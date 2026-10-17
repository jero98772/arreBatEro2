---
name: torx
description: Write correct, idiomatic Torx code. Torx is Extropic's JAX library for parametrised stochastic circuits - probabilistic gates on pbits, pdits and pmodes (continuous Gaussian sites) composed into DiscretePCircuit or HybridPCircuit and run on StateVectorSimulator, BranchingSimulator or AffineGaussianSimulator, all built on a directed factor graph (Site, DFG, TiledFactor, ChainFactor). Use this skill whenever the user mentions Torx or torx.psc, Extropic stochastic circuits, probabilistic gates such as PNOT, PCNOT, PSWAP, PJUMP, PISING, PditCycle, AffineGaussianGate or MixtureGaussianGate, pbits, pdits or pmodes, parameter-shift gradients through a sampler, Trotterized random walks or chemical reaction networks as circuits, Gaussian state-space models built from gates, or custom factors and directed factor graphs in JAX, even if they do not say Torx by name. For plain block Gibbs sampling or training an Ising or Boltzmann EBM, use the thrml skill instead.
---

# Torx

Torx is Extropic's JAX library for **parametrised stochastic circuits (PSCs)**: ordered
lists of probabilistic gates that reshape probability distributions. A gate is a
stochastic kernel acting on one of three data primitives:

- **pbit** - binary site, values `{0, 1}`
- **pdit** - `d`-state site, values `{0, ..., d-1}`
- **pmode** - continuous site, values in `R^N`, tracked by mean and covariance

Every gate is also a *factor*, every circuit is a *directed factor graph (DFG)*, and a
*simulator* compiles a circuit together with its parameters and reads it back as
samples, an exact density, or exact Gaussian moments. Everything is JAX: it works
with `jit`, `vmap`, `grad`, with the caveat that gates, circuits and DFGs are Equinox
modules holding static structure, while their parameters live outside them (see
"Static structure vs. traced arrays").

When implementing something, write the code. No emojis. When you explain a concept,
tie it to the concrete Torx object that realizes it.

## Torx or thrml?

| Task | Library |
|------|---------|
| Block Gibbs sampling, training an Ising / RBM / Boltzmann EBM by contrastive divergence | **thrml** |
| Stochastic gate circuits, Markov kernels, exact densities, parameter-shift gradients, hybrid discrete-continuous processes, Gaussian moment propagation and conditioning, custom directed factor graphs | **Torx** |

The Torx gallery itself says Gibbs sampling "is not Torx's main use case; thrml is the
dedicated library". If a user still wants Gibbs in Torx (for instance to exercise the
DFG machinery), follow the factor-graph Gibbs recipe in `references/examples.md`.

## Import rule

Gates, circuits and simulators come from `torx.psc`. Every gallery notebook imports
this way, and the API docs cross-reference symbols as `torx.psc.X`:

```python
from jax.scipy.special import logit          # gates take logits, not probabilities
from torx.psc import DiscretePCircuit, HybridPCircuit
from torx.psc import PNOT, PCNOT, PSWAP, PJUMP, PReset, PISING, PditShift, PditCycle
from torx.psc import AffineGaussianGate, Displace, Scale, Mix, Diffuse, MixtureGaussianGate
from torx.psc import StateVectorSimulator, BranchingSimulator, AffineGaussianSimulator
```

The factor-graph core (`AbstractFactor`, `AbstractReferenceFactor`,
`AbstractHasLogProbability`, `AbstractMatrixFactor`, `Site`, `DFG`, `DFGInfo`,
`TiledFactor`, `ChainFactor`, `DeterministicFactor`), `draw_circuit`, and the sampler
providers (`AbstractSampler`, `JaxPRNGSampler`) are public, but the docs never print
their module path. **Do not guess a deep path.** Run
`python scripts/locate_symbols.py Site DFG TiledFactor ...` against the installed
package and import from the shallowest module it reports. If you cannot run code,
write `from torx import ...` (the docs list these as "public torx symbols") and tell
the user in one line that the import path is unverified.

Torx needs JAX and Equinox; training examples add `optax`, references use
`numpy`/`scipy`, graph examples use `networkx`. Torx may be private: install it the
way the user's environment provides, never `pip install torx` on a guess.

## How the API composes

| Layer | Object | What you supply |
|-------|--------|-----------------|
| Site | integer site ids | discrete ids and continuous ids are **separate namespaces** |
| Gate | `PNOT(0)`, `PCNOT([0, 1])`, `AffineGaussianGate(sites=[0, 1], dims=(1, 1))` | which sites it touches, nothing else |
| Parameters | `thetas`: a list with one pytree per gate, aligned with `circuit.gates` | the numbers |
| Circuit | `DiscretePCircuit(gates, reps=m)`, `HybridPCircuit(gates, reps=m)` | gate order and repetition count |
| Simulator | `compiled = sim.build_circuit(circuit, thetas)`, then `density` / `sample` / `expval_all` / `propagate` / `condition` | initial state and key |
| Factor graph | `Site`, `DFG`, `TiledFactor`, `ChainFactor`, custom factor subclasses | conditionals and wiring, when a gate list is the wrong shape |

Every object from a gate up to a DFG honours one contract:
`sample(key, inputs, params, info=None, site_info=None, return_aux=False)`, plus
`sample_multiple(..., n_samples=N)` (vmapped over `N` split keys) and
`init_params(key)`. The layers nest, so you only drop to factors for models a gate
list cannot express.

## The one rule: gates are structure, thetas are data

This is the single most important Torx idiom. A gate stores its sites (and `dims`),
never its parameters. Parameters live in a separate list aligned with the gate order:

```python
circuit = DiscretePCircuit([PNOT(0), PCNOT([0, 1]), PNOT(1)])
thetas = [jnp.array([logit(0.38)]), jnp.array([logit(0.70)]), jnp.array([logit(0.25)])]
assert len(thetas) == len(circuit.gates)
defaults = circuit.init_params(jax.random.key(0))     # same structure, default values
```

What follows from it:

- `reps=m` replays the whole gate list `m` times with the **same** thetas, which is how
  Trotter layers and repeated sweeps are written.
- Changing values never means rebuilding the circuit: compile once and swap thetas
  with `eqx.tree_at` (see below).
- To share one *learnable* parameter across many gates, keep a small shared vector
  and broadcast it with an index map (`shared[param_map][:, None]`); autodiff then
  accumulates every gate's gradient into the shared entry. Putting the same theta
  object into the list several times ties values at inference time but does **not**
  tie gradients.

## Reading a gate: parameter forms and conventions

| Gates | Theta | Meaning |
|-------|-------|---------|
| `PNOT(i)`, `PReset(i)`; `PCNOT`, `PSWAP`, `PJUMP`, `PCopy`, `POR`, `PDEMUX` on `[a, b]`; `PMultiCNOT([c1, ..., t])`; `PCSWAP([c, a, b])`; `PditShift(i, dims=d)`; `PditSWAP([a, b], dims=d)` | shape `(1,)` logit | operation branch with probability `sigmoid(theta)`. Pass `jnp.array([logit(p)])`, never `p` |
| `PditCycle(sites=i, dims=d)` | shape `(2,)` | `[stay, forward, backward] = softmax([0, t0, t1])`, so `theta = log([p_fwd / p_stay, p_back / p_stay])` |
| `PISING([i, j])` | shape `(5,)` = `[J, h1, h2, beta, dt]` | Glauber bond kernel `exp(dt * Q)`; all zeros is the identity |
| `AffineGaussianGate(sites, dims)` | dict `A (d,d)`, `b (d,)`, `log_var (d,)` | `x' = A x + b + N(0, diag(exp(log_var)))`; default is identity plus unit noise |
| `Displace` / `Scale` | `(d,)` | `x + theta` / `x * exp(theta)` |
| `Mix(sites=[a, b], dims=(1, 1))` | scalar angle | rotates the two scalar sites |
| `Diffuse`, `GaussianNoiseGate` | `(d,)` log-variance | `x + N(0, exp(theta))`; Brownian `D` over `dt` is `theta = log(2 D dt)` |
| `MixtureGaussianGate(sites=(disc, cont), dims=(d,), num_components=K)` | dict `means (K,d)`, `log_vars (K,d)` | `x + mu_k + N(0, diag(sigma_k^2))` where `k` is the control value |
| `JumpDiffusionGate(sites=(disc, cont), dims)` | dict `diff_log_var`, `jump_mean`, `jump_log_var` | diffusion always, plus a Gaussian jump when the control is 1 |

`references/api.md` has every gate's branches and every constructor signature.

Four conventions behind most silent bugs:

1. **`get_matrix(theta)` is column-stochastic `P[out, in]`**; columns sum to 1. The
   factor-side `get_log_probability_matrix` is the transpose's log (row = input).
   Check `M.sum(axis=0)` before trusting a hand-built kernel.
2. **Basis order is big-endian**: site 0 is the most significant digit. For two pbits
   the flat index is `2 * s0 + s1`, ordering `00, 01, 10, 11`; in general
   `np.ravel_multi_index(samples.T, dims)`.
3. **Branch 0 is usually, not always, the identity.** `PDEMUX` branch 0 resets the
   work pbit. Read `gate.branches` (shape `(K, basis_size, num_sites)`) when unsure.
4. **A finite-logit `PReset` leaks** (`sigmoid(theta) < 1`). For an exact zero, start
   the site in state 0 instead of resetting it.

Mixture weights are **not** in `MixtureGaussianGate`'s theta. They come from the
distribution of the discrete control site, usually set by a `PditShift` (K = 2:
`theta = log(w1 / w0)`) or `PditCycle` (K = 3: `theta = log(pi[1:] / pi[0])`) placed
before it, starting from control 0. Hybrid gates currently accept a single discrete
control site.

## Picking a simulator

| Simulator | Circuit | Initial state | Returns | Gates | Limit |
|-----------|---------|---------------|---------|-------|-------|
| `StateVectorSimulator()` | `DiscretePCircuit` | **float** probability vector, length `prod(dims)` (one-hot for a basis start) | `density` (flat, exact), `expval(c, x, site)`, `expval_all` - exact and autodiff-differentiable | every discrete gate, including `PISING` | memory is `prod(dims)`; plan for at most about `2**16`-`2**20` states |
| `BranchingSimulator(diff_method=..., num_samples=N)` | `DiscretePCircuit` | **int** basis state, shape `(num_sites,)` | `sample` -> `(N, num_sites)` ints; `expval`, `expval_all` Monte Carlo estimates, differentiable by parameter shift | branch lookup-table gates only (the `AbstractKBranchGate` family) | **no `PISING`** or other generator gates |
| `AffineGaussianSimulator()` | `HybridPCircuit` of affine-Gaussian gates | flat float continuous vector | `propagate` -> `GaussianMoments`; `condition` (Schur complement); `expval`, `expval_all` | `AffineGaussianGate`, `Displace`, `Scale`, `Mix`, `Diffuse`, `GaussianNoiseGate`, custom `AbstractAffineGaussianGate` | dense `O(D^3)` per gate; no mixtures or nonlinear gates |
| none: `circuit.sample` / `sample_multiple` | `HybridPCircuit` | `{"discrete": int array, "continuous": float array}` | a sampled state dict with the same keys | everything, including custom continuous gates | samples only |

For `expval` on a pdit, Torx returns the expected site value; for a pbit that equals
`P(site = 1)`. A `HybridPCircuit` with only continuous gates still needs a discrete
entry: pass `jnp.zeros(0, jnp.int32)`.

## Static structure vs. traced arrays

**Structure is static, arrays are dynamic.** Gates, circuits, `Site`s and `DFG`s are
fixed at trace time; thetas, initial states, inputs and keys flow through
`jit`/`vmap`/`grad`.

- `eqx.filter_jit`, `eqx.filter_vmap`, `eqx.filter_value_and_grad` treat non-array
  leaves as static, so you can pass whole circuits, compiled circuits and models as
  arguments. Under plain `jax.jit`, close over the structure and pass arrays only.
- **Compile once, swap thetas.** The compiled object is a pytree, so replace its
  `thetas` leaf instead of rebuilding:

  ```python
  template = sim.build_circuit(circuit, [jnp.zeros((1,)) for _ in circuit.gates])

  def with_logits(rows):                          # rows: (num_gates, max_branches - 1)
      return eqx.tree_at(lambda c: c.thetas, template, rows)
  ```

  For `BranchingSimulator`, `compiled.thetas` is one array of shape
  `(num_gates, max_branches - 1)` **padded with `-inf`** for gates with fewer
  branches; keep that padding when you swap. For `StateVectorSimulator` and
  `AffineGaussianSimulator` it is a list aligned with the gates. Calling
  `build_circuit` inside a traced function also works; it just redoes the lowering
  each trace.
- **Keys are arrays, thread them.** Use typed keys (`jax.random.key(seed)`), split
  before you branch, give every chain, sample and time step its own key, and use
  `jax.random.fold_in` to derive per-case keys. Composites split internally:
  `TiledFactor` is a vmap over `jax.random.split(key, n_tiles)` and `ChainFactor` a
  loop over `jax.random.split(key, n_steps)`, so a manual re-implementation with the
  same keys reproduces their draws exactly.
- **Know what is differentiable.** `StateVectorSimulator.density`/`expval*` and
  `AffineGaussianSimulator` moments (with finite `log_var`) are exact JAX functions.
  `BranchingSimulator.expval*` is differentiable through its `diff_method`.
  `BranchingSimulator.sample` returns integers and has no gradient. Do not rely on
  pathwise gradients through `HybridPCircuit.sample`; the docs do not promise them.
- **Loops**: `reps` for identical layers; `jax.lax.scan` over `circuit.sample` when
  you need a per-step readout; `vmap` over independent chains; jit the outer function
  once (pass temperatures and similar scalars as traced arrays so changing them does
  not recompile).

## Sampling a discrete circuit

The canonical path, exact and sampled side by side:

```python
import jax, jax.numpy as jnp
from jax.scipy.special import logit
from torx.psc import DiscretePCircuit, PNOT, PCNOT, StateVectorSimulator, BranchingSimulator

circuit = DiscretePCircuit([PNOT(0), PCNOT([0, 1]), PNOT(1)])
thetas = [jnp.array([logit(0.38)]), jnp.array([logit(0.70)]), jnp.array([logit(0.25)])]
start = jnp.array([0, 0], jnp.int32)                     # basis state |00)

# Exact: the input is a distribution over prod(dims) = 4 states.
sv = StateVectorSimulator()
exact = sv.build_circuit(circuit, thetas)
x0 = jnp.zeros(4).at[2 * start[0] + start[1]].set(1.0)   # derive one-hot from the bits
density = sv.density(exact, x0)                          # (4,) over 00, 01, 10, 11
means = sv.expval_all(exact, x0)                         # (2,) = P(s_i = 1)

# Sampled: the input is an int basis state; the output is raw bitstrings.
bs = BranchingSimulator(num_samples=5_000)
compiled = bs.build_circuit(circuit, thetas)
samples = bs.sample(compiled, start, jax.random.key(0))  # (5000, 2) int
idx = samples[:, 0] * 2 + samples[:, 1]
empirical = jnp.bincount(idx, length=4) / samples.shape[0]
sample_mean = samples.mean(axis=0)                       # estimates `means`
```

Histograms, means and confidence intervals are ordinary array reductions over
`samples` rows; use `jnp` when the readout must be jit- or grad-compatible. Sampled
estimates concentrate at `1 / sqrt(N)`: the binomial standard error is at most
`0.5 / sqrt(N)`.

## Continuous and hybrid circuits

Affine-Gaussian stacks have exact moments; any hybrid circuit can be sampled
directly:

```python
from torx.psc import HybridPCircuit, Diffuse, Displace, Scale, Mix, AffineGaussianSimulator

step = HybridPCircuit(
    [Diffuse(sites=0, dims=(1,)), Displace(sites=0, dims=(1,)),
     Scale(sites=0, dims=(1,)), Mix(sites=[0, 1], dims=(1, 1))],
    reps=4,
)
thetas = [jnp.log(jnp.array([2 * 0.25 * 0.4])),   # Diffuse: log(2 D dt)
          jnp.array([0.17]),                        # Displace
          jnp.log(jnp.array([1.10])),               # Scale: log-scale
          jnp.asarray(0.35)]                        # Mix: scalar angle

ag = AffineGaussianSimulator()
compiled = ag.build_circuit(step, thetas)
prior = ag.propagate(compiled, jnp.zeros(2))       # GaussianMoments: .mean, .covariance
post = ag.condition(compiled, observations={1: jnp.array([0.35])},
                    initial_continuous=jnp.zeros(2), query_sites=[0])

init = {"discrete": jnp.zeros(0, jnp.int32), "continuous": jnp.zeros(2)}
draws = step.sample_multiple(jax.random.key(0), init, thetas, n_samples=20_000)["continuous"]
```

The set of observed and queried sites in `condition` must be static Python (dict keys,
lists); only observation values and `jitter` may be traced.

**Write-to-a-new-site idiom.** A multi-site `AffineGaussianGate` maps the concatenated
local state of its sites. To read site `a` and write site `b` while leaving `a`
untouched, use the block matrix `[[I, 0], [M, 0]]` with `log_var = -inf` on `a`'s
coordinates (an exact deterministic copy). Unrolled state-space models and Kalman /
RTS smoothers are built this way (`references/examples.md`). `-inf` is exact for
conditioning but not differentiable: clamp it to a large negative finite value before
training.

For discrete control of continuous sites, put the control gate first:
`HybridPCircuit([PditCycle(sites=0, dims=K), MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=K)])`,
then `jax.lax.scan` over `one_step.sample` for trajectories (regime-switching
diffusion recipe in `references/examples.md`).

## Training a circuit

There are three gradient routes. Pick by circuit size and objective:

1. **Exact (small circuits).** Build the thetas from traced parameters inside the loss,
   compile with `StateVectorSimulator`, and `jax.grad` through `density` or `expval`.
2. **Sampled expectations (large circuits).** `BranchingSimulator(diff_method=...)` and
   grad through `expval` / `expval_all`. For a two-branch gate
   `G = (1 - s) I + s B` with `s = sigmoid(theta)`, the rule is
   `dE[O]/dtheta = s (1 - s) (E[O | B] - E[O | I])`, evaluated with the gate clamped to
   each branch. `"param_shift_inf"` (default) clamps every parameter to
   `theta = +-inf`, so cost grows linearly with the number of parameters;
   `"param_shift_single"` reuses the primal run to save evaluations;
   `"param_shift_filter"` estimates all gradients from one forward pass by
   partitioning samples on the branch each gate took (cheapest, noisiest).
3. **Discrete objectives on whole bitstrings** (cut values, rewards). Use a
   score-function (REINFORCE) estimator on exact log-probabilities:
   sample indices with `jax.random.categorical(key, stop_gradient(logp))`, weight
   `logp[idx]` by `stop_gradient(r - r.mean())`, and multiply by `M / (M - 1)` to
   remove the same-batch-baseline bias (MaxCut recipe in `references/examples.md`).

Route 2 with optax:

```python
import equinox as eqx, optax

sim = BranchingSimulator(diff_method="param_shift_inf", num_samples=256)
template = sim.build_circuit(circuit, [jnp.zeros((1,)) for _ in circuit.gates])
start = jnp.array([0, 0], jnp.int32)
target = jnp.array([0.2, 0.9])

def loss(logits, key):                                   # logits: (num_gates, 1)
    compiled = eqx.tree_at(lambda c: c.thetas, template, logits)
    return jnp.sum((sim.expval_all(compiled, start, key) - target) ** 2)

optimizer = optax.adam(5e-2)
logits = jnp.zeros((len(circuit.gates), 1))
opt_state = optimizer.init(logits)

@jax.jit
def train_step(logits, opt_state, key):
    value, grads = jax.value_and_grad(loss)(logits, key)
    updates, opt_state = optimizer.update(grads, opt_state, logits)
    return optax.apply_updates(logits, updates), opt_state, value
```

Sign convention: `optimizer.update` negates internally and `optax.apply_updates` adds,
so pass the gradient of the loss you **minimize**. To maximize an objective, minimize
its negative. Use a fresh key every step.

### A worked wrapper: a shared-kernel model

The gallery's stochastic CNN is the shape to copy for any trainable Torx model: the
module holds the trainable arrays plus *integer* structure arrays, and each call swaps
the broadcast logits into a precompiled template.

```python
# template = sim.build_circuit(circuit, [jnp.zeros((1,)) for _ in circuit.gates]), built once
class SCNN(eqx.Module):
    gate_logits: jax.Array          # (5,) shared logits, trained
    linear: eqx.nn.Linear           # readout head, trained
    _param_map: jax.Array           # (60,) int: gate -> shared logit; skipped by the optimizer
    _readout_sites: jax.Array       # (4,) int

    def __call__(self, x, key, sampler):
        full = self.gate_logits[self._param_map][:, None]          # (60, 1)
        compiled = eqx.tree_at(lambda c: c.thetas, template, full)
        features = sampler.expval_all(compiled, x, key)[self._readout_sites]
        return jax.nn.sigmoid(jnp.squeeze(self.linear(features)))

optimizer = optax.adam(5e-2)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))   # ints excluded
```

Integer arrays are not inexact, so `eqx.filter(model, eqx.is_inexact_array)` keeps them
away from the optimizer, and the fancy-indexing broadcast is what accumulates gradients
from all 60 physical gates into 5 logits.

## Going lower-level: factors and directed factor graphs

Reach below circuits only when the model is not a fixed gate list: a custom
conditional, deterministic glue between stages, or explicit graph wiring.

**A custom factor** subclasses `AbstractReferenceFactor` (which supplies
`sample_with_references`), declares its ports as static `jax.ShapeDtypeStruct`s, and
implements `init_params` and `sample`:

```python
BIT = jax.ShapeDtypeStruct((), jnp.int32)
DRIVE = jax.ShapeDtypeStruct((), jnp.float32)

class ConditionalBit(AbstractReferenceFactor):
    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports = {"drive": DRIVE}
        self.output_spec = BIT

    def init_params(self, key):
        return {"w": jnp.array(2.0), "b": jnp.array(-1.0)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        p = jax.nn.sigmoid(params["w"] * inputs["drive"] + params["b"])
        out = jax.random.bernoulli(key, p).astype(jnp.int32)
        return (out, None) if return_aux else out
```

Add `AbstractHasLogProbability` and implement
`log_probability(inputs, outputs, params, ...)` when the conditional is analytic; use
`AbstractMatrixFactor` (`input_states`, `output_states`,
`get_log_probability_matrix` with row = input) for finite tables.

**Wiring a DFG.** `Site(name, factor, parents, porting_fn, param_key, info_key, site_info)`
places a factor. `parents` are addresses (DFG input ports or site names, which must be
disjoint); `porting_fn` is either a tuple of port names matching `parents` one to one,
or a callable that receives the parent outputs as a tuple and returns the factor's
input dict. Sites sharing a `param_key` share parameters (weight tying); use `None`
for parameterless sites.

```python
# CoinFactor follows the same pattern with input_ports = {} and output_spec = BIT.
def to_drive(parent_outputs):
    return {"drive": parent_outputs[0].astype(jnp.float32)}

sites = (
    Site("coin", CoinFactor(), (), (), "src", info_key=None, site_info=None),
    Site("out", ConditionalBit(), ("coin",), to_drive, "cond", info_key=None, site_info=None),
)
graph = DFG(sites, {}, "out")                       # (sites, input_ports, output_name)
params = graph.init_params(jax.random.key(0))       # {"src": ..., "cond": ...}
y = graph.sample(jax.random.key(1), {}, params)     # same contract as one factor
```

**Composites.** `TiledFactor(base, n_tiles, weight_tied=True)` runs `n_tiles` copies in
parallel: every input gains a leading `n_tiles` axis and so does the output.
`ChainFactor(base, n_steps, feedback_porting_fn="state", weight_tied=True)` runs a
`lax.scan`, feeding each step's output into the named port (other ports are broadcast
unchanged) and returning the final output; the fed-back output must match that port's
spec, int to int. A DFG is itself a factor, so a whole sweep can be the base of a
`ChainFactor`. `DeterministicFactor(fn, input_ports, output_spec)` wraps
`fn(inputs, site_info)` for reassembly glue. Pass
`DFGInfo(expose_site_outputs=True)` as `info` with `return_aux=True` to read every
site's output.

### Writing a correct custom gate

For a continuous update with no closed form (Langevin, MALA, any nonlinear drift),
subclass `AbstractContinuousGate`: the circuit passes the gate's theta as `params`
and the local substate as `inputs["continuous"]`; return the new continuous values.
If the update *is* affine-Gaussian, subclass `AbstractAffineGaussianGate` and implement
`affine_parameters(theta) -> (A, b, log_var)` instead; you then get exact moments and
conditioning from `AffineGaussianSimulator` for free. Rules for either:

- **Branch on static configuration, never on values.** Store switches like a
  Metropolis flag as `eqx.field(static=True)`; a Python `if` on them is fine. Inside
  the update use `jnp.where` / `lax.cond`.
- Derive drifts with `jax.grad` of an `energy` method on the gate, so the drift and the
  target `exp(-V / T)` are the same potential.
- Use stable primitives (`jax.nn.log_sigmoid`, `logsumexp`, `jax.random.categorical`
  on logits) and return `(out, None)` when `return_aux=True`.

The full Langevin / MALA template is in `references/examples.md`.

## Testing what you build

Stochastic code fails quietly: a transposed kernel or a probability passed as a logit
still produces plausible samples. Validate against ground truth, not by eye:

- **Exact references.** Up to roughly `2**16` states, enumerate with
  `StateVectorSimulator.density` or brute force. For continuous-time Markov chains
  compare with `scipy.linalg.expm(Q t) @ p0`; for Gaussian circuits, with
  `AffineGaussianSimulator` or a dense NumPy recursion plus Schur complement.
- **Separate the error sources.** Compare a sampled result first with the exact mean of
  the *same discretized* scheme (isolates Monte Carlo error), then that mean with the
  continuous-time answer (isolates Trotter / Euler bias). First-order splitting bias
  falls as `1 / m`; a symmetric Strang sweep as `1 / m**2`.
- **Tolerances scale with N.** Use `k * SE` with `SE = 0.5 / sqrt(N)` for pbit means,
  `5 / sqrt(N)` for Kolmogorov-Smirnov distances, and per-moment standard errors for
  Gaussian means and covariances.
- **Kernel sanity.** `get_matrix(theta).sum(axis=0)` is 1; `exp` of a matrix factor's
  log-probability rows sums to 1.
- **Same-key equivalence.** A `TiledFactor` must equal a manual vmap over
  `jax.random.split(key, n_tiles)`, and a `ChainFactor` a manual loop over
  `jax.random.split(key, n_steps)`; any difference is a wiring bug.
- **Gradient check.** Compare a parameter-shift gradient with a central finite
  difference on the same batch and key (about 30% relative tolerance for sampled
  estimates, tighter with more samples), or with `jax.grad` of the exact
  `StateVectorSimulator` loss on a tiny circuit.

Assert these checks in code rather than printing them, so a drift fails loudly.

## Quick gotcha list

- Probability passed where a logit belongs; scalar passed where shape `(1,)` belongs
  (only `Mix` takes a scalar).
- `PISING` on `BranchingSimulator` (unsupported); int input to `StateVectorSimulator`,
  float distribution to `BranchingSimulator` (swapped).
- Per-edge `PISING` sweeps omit each endpoint's outside neighbors, so they do **not**
  sample the full Boltzmann law of a graph; a one-site conditional with the full local
  field does.
- Gate order matters inside a layer: the stochastic matrices do not commute.
- `-inf` padding lost when hand-building `BranchingSimulator` thetas for mixed-branch
  circuits; `-inf` log-variances left in a model you differentiate.
- Treating aliased thetas in a list as gradient tying.
- Reusing a key; deriving many keys from one split.

## References

- `references/api.md` - annotated API: every class and gate with constructor,
  theta shape, branches and simulator support. Ground truth for signatures.
- `references/examples.md` - worked recipes distilled from the Torx gallery: graph
  heat flow and Strang splitting, directed transport and chemical reaction networks,
  readouts and error bars, Gibbs updates, posterior per-pixel sampling, stochastic CNN,
  MaxCut with REINFORCE, Gaussian state-space smoothing, regime-switching diffusion,
  Gaussian mixtures, a custom Langevin gate, and a factor-graph Gibbs sweep.
- `references/jax_equinox.md` - the static-versus-traced, key, gradient and testing
  rules as a checklist. Load it when writing or reviewing Torx code that trains a
  model or drops below the circuit layer.
- `scripts/locate_symbols.py` - prints which installed `torx` module exports each
  name, with its signature. Run it before importing anything outside `torx.psc`.
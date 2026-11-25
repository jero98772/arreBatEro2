"""
Hybrid pipeline: one matrix flows through four backends.

    torx       C = A @ B          (stochastic affine-Gaussian gate, exact moments)
      |
    thrml      M = <s_i s_j>      (Ising model with couplings built from C, block Gibbs sampling)
      |
    PennyLane  Q = <Z_k>          (each row of M is angle-encoded into a 4-qubit circuit)
      |
    Q.ANT      Y = Q @ W.T        (photonic matmul in bfloat16 via the bridge: NPU, simulator
                                   or NumPy emulation, whichever is available)

Every stage is checked against an exact reference. JAX (float32) is the hub; each
backend converts only at its own boundary.

Tested with: Python 3.13, jax 0.7.2, extro-torx 0.0.3, thrml 0.1.4, pennylane 0.45.1,
qant_native_computing_toolkit 2.5.0 (cpu-backend build) -- optional: without it,
the Q.ANT bridge below emulates the NPU in NumPy.

Q.ANT backend choice (override with QANT_BACKEND=npu|sim|emulate):
    npu      real Q.ANT hardware
    sim      toolkit built with `-F cpu-backend` (same API, runs on the CPU)
    emulate  toolkit not installed: NumPy reproduces linear_fprop's bfloat16 behaviour
"""

import itertools
import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from ml_dtypes import bfloat16

from torx.psc import AffineGaussianGate, AffineGaussianSimulator, HybridPCircuit
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# ===========================================================================
# Q.ANT bridge: photonic matmul callable from JAX, with or without an NPU
# ===========================================================================
try:
    from qant_native_computing_toolkit import generic, info, native, utils
    from qant_native_computing_toolkit.generic import DataRole, QantDeviceCapability

    _HAVE_TOOLKIT = True
except ImportError:  # toolkit (or its driver) missing -> emulation only
    _HAVE_TOOLKIT = False


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Backend:
    kind: str  # "npu" | "sim" | "emulate"
    device_id: int = -1  # -1 lets the toolkit choose
    serial: str = "numpy-emulation"
    capabilities: tuple = field(default_factory=tuple)

    def __str__(self) -> str:
        caps = ", ".join(self.capabilities) or "linear only"
        return f"{self.kind} (device {self.device_id}, {self.serial}; {caps})"


def detect_backend(prefer: str | None = None) -> Backend:
    """Find the best backend for matrix multiplication.

    Order: real NPU, then the cpu-backend simulator, then NumPy emulation.
    `prefer` (or the QANT_BACKEND env var) forces one of them.
    """
    prefer = prefer or os.environ.get("QANT_BACKEND")
    if prefer == "emulate" or not _HAVE_TOOLKIT:
        if prefer in ("npu", "sim"):
            raise RuntimeError(
                f"QANT_BACKEND={prefer} but the Q.ANT toolkit is not installed."
            )
        return Backend("emulate")

    needed = [QantDeviceCapability.Mul, QantDeviceCapability.MemHandling]
    npus = generic.get_available_npus(needed)  # [(device_id, serial), ...]
    hardware = [(d, s) for d, s in npus if not s.startswith("cpu_backend")]
    simulated = [(d, s) for d, s in npus if s.startswith("cpu_backend")]

    if prefer == "npu" and not hardware:
        raise RuntimeError("QANT_BACKEND=npu but no Q.ANT NPU was found.")
    if prefer == "sim" and not simulated:
        raise RuntimeError(
            "QANT_BACKEND=sim but the toolkit is not the cpu-backend build."
        )

    if hardware and prefer != "sim":
        d, s = hardware[0]
        kind = "npu"
    elif simulated:
        d, s = simulated[0]
        kind = "sim"
    else:
        return Backend("emulate")

    caps = tuple(
        str(c).rsplit(".", 1)[-1] for c in info.get_capabilities(d)
    )  # Rust enum, no .name
    return Backend(kind, d, s, caps)


# ---------------------------------------------------------------------------
# Emulation of linear_fprop (used when there is no toolkit)
# ---------------------------------------------------------------------------
def _emulated_linear_fprop(x_bf16: np.ndarray, w_bf16: np.ndarray) -> np.ndarray:
    """features @ weights.T with bfloat16 inputs and output, float32 accumulate.

    Matches the cpu-backend simulator to within one bfloat16 rounding step.
    It does NOT model analog hardware error; use `noise_std` in training for that.
    """
    y = x_bf16.astype(np.float32) @ w_bf16.astype(np.float32).T
    return y.astype(bfloat16)


def _to_bf16(a, align: bool) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(a, dtype=np.float32)).astype(bfloat16)
    if align and _HAVE_TOOLKIT:
        out = utils.align_ndarray_page_boundary(out)
    return out


# ---------------------------------------------------------------------------
# Training path: hardware-aware, pure JAX, no device needed
# ---------------------------------------------------------------------------
def _bf16_ste(a):
    """Round to bfloat16 in the forward pass, identity gradient (straight-through)."""
    return a + jax.lax.stop_gradient(a.astype(jnp.bfloat16).astype(a.dtype) - a)


def hw_aware_linear(x, w, b=None, *, key=None, noise_std: float = 0.0):
    """x @ w.T as the NPU will compute it, but differentiable.

    - inputs, weights and output are rounded to bfloat16 (straight-through gradient)
    - optional Gaussian noise on the output, relative to its typical size, so the
      trained model tolerates analog error (Q.ANT's own examples use std ~0.04)
    """
    y = _bf16_ste(_bf16_ste(x) @ _bf16_ste(w).T)
    if key is not None and noise_std > 0:
        scale = jax.lax.stop_gradient(jnp.std(y) + 1e-6)
        y = y + noise_std * scale * jax.random.normal(key, y.shape, y.dtype)
    if b is not None:
        y = y + b
    return y


# ---------------------------------------------------------------------------
# Deployment path: weights resident on the device
# ---------------------------------------------------------------------------
class QantLinear:
    """y = x @ W.T (+ b) on the Q.ANT backend, callable from JAX.

    Weights are converted and uploaded once. Calls work under jax.jit and
    jax.vmap, and jax.grad flows to the input x (the gradient uses the float32
    weights, because the device has no backward pass). To train W itself, use
    `hw_aware_linear`, then build a new QantLinear or call `set_weights`.
    """

    def __init__(self, weight, bias=None, backend: Backend | str | None = None):
        self.backend = (
            backend if isinstance(backend, Backend) else detect_backend(backend)
        )
        weight = jnp.asarray(weight, jnp.float32)
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D (n_out, n_in), got {weight.shape}")
        self.n_out, self.n_in = weight.shape
        self.bias = None if bias is None else jnp.asarray(bias, jnp.float32)
        self._handle = None
        self.set_weights(weight)
        self._fn = self._build_jax_fn()

    # -- weights ------------------------------------------------------------
    def set_weights(self, weight) -> None:
        """Replace the weights. On a device, reuses the existing memory block."""
        weight = jnp.asarray(weight, jnp.float32)
        if weight.shape != (self.n_out, self.n_in):
            raise ValueError(f"expected {(self.n_out, self.n_in)}, got {weight.shape}")
        self.weight = weight
        self._w_bf16 = _to_bf16(weight, align=True)
        if self.backend.kind == "emulate":
            return
        if self._handle is None:
            self._handle = generic.upload(
                self._w_bf16, DataRole.Weights, device_id=self.backend.device_id
            )
        else:
            generic.upload_into(self._w_bf16, self._handle)

    # -- host-side forward (NumPy in, NumPy out) ----------------------------
    def forward_numpy(self, x) -> np.ndarray:
        x_bf = _to_bf16(x, align=True)
        if x_bf.shape[-1] != self.n_in:
            raise ValueError(f"last dim of x must be {self.n_in}, got {x_bf.shape}")
        if self.backend.kind == "emulate":
            y = _emulated_linear_fprop(x_bf, self._w_bf16)
        else:
            y = native.linear_fprop_preloaded_weights(x_bf, self._handle)
        return y.astype(np.float32)

    # -- JAX wrapper ----------------------------------------------------------
    def _build_jax_fn(self):
        layer = self

        @jax.custom_vjp
        def f(x):
            out = jax.ShapeDtypeStruct(x.shape[:-1] + (layer.n_out,), jnp.float32)
            return jax.pure_callback(
                layer.forward_numpy, out, x, vmap_method="expand_dims"
            )

        def f_fwd(x):
            return f(x), None

        def f_bwd(_, g):
            return (g @ layer.weight,)

        f.defvjp(f_fwd, f_bwd)
        return f

    def __call__(self, x):
        y = self._fn(jnp.asarray(x, jnp.float32))
        return y if self.bias is None else y + self.bias

    # -- diagnostics ----------------------------------------------------------
    def report(self) -> str:
        lines = [
            f"Q.ANT backend: {self.backend}",
            f"layer: {self.n_in} -> {self.n_out}",
        ]
        if self.backend.kind == "npu":
            pc = info.get_perf_counter(self.backend.device_id)
            busy = 1 - pc["stalling_counter"] / max(pc["timebased_counter"], 1e-12)
            lines.append(
                f"NPU busy fraction: {busy:.1%} (low = host is the bottleneck)"
            )
            lines.append(f"sensors: {info.get_sensor_info(self.backend.device_id)}")
        return "\n".join(lines)


# ===========================================================================
# Pipeline
# ===========================================================================
N = 4  # matrix size = number of spins = number of qubits
key = jax.random.key(0)
k_a, k_b, k_w, k_init, k_run = jax.random.split(key, 5)


# ---------------------------------------------------------------------------
# Stage 1: torx -- matrix multiplication as a stochastic circuit
# ---------------------------------------------------------------------------
# AffineGaussianGate computes x' = A x + b + N(0, exp(log_var)) on one continuous
# site of dimension N. Propagating each column of B through it gives A @ B
# (mean) plus the uncertainty the noise adds (covariance).
def torx_matmul(A, B, log_var=-6.0):
    circuit = HybridPCircuit([AffineGaussianGate(sites=[0], dims=(N,))])
    theta = [{"A": A, "b": jnp.zeros(N), "log_var": jnp.full(N, log_var)}]
    sim = AffineGaussianSimulator()
    compiled = sim.build_circuit(circuit, theta)

    def one_column(col):
        moments = sim.propagate(compiled, col)
        return moments.mean, jnp.sqrt(jnp.diag(moments.covariance))

    mean, std = jax.vmap(one_column, in_axes=1, out_axes=1)(B)
    return mean, std


A = jax.random.normal(k_a, (N, N)) * 0.7
B = jax.random.normal(k_b, (N, N)) * 0.7
C, C_std = torx_matmul(A, B)

err_torx = float(jnp.abs(C - A @ B).max())
assert err_torx < 1e-5, err_torx
print("Stage 1  torx       C = A @ B")
print(np.round(np.asarray(C), 3))
print(
    f"         max |C - A@B| = {err_torx:.1e}, noise std per entry = {float(C_std.mean()):.3f}\n"
)


# ---------------------------------------------------------------------------
# Stage 2: thrml -- C becomes an Ising model, sampled with block Gibbs
# ---------------------------------------------------------------------------
# Couplings J_ij come from the symmetric part of C, biases from its diagonal,
# both scaled to a range where the sampler mixes well.
def ising_from_matrix(C):
    J = (C + C.T) / 2
    off = [(i, j) for i in range(N) for j in range(i + 1, N)]
    w = jnp.array([J[i, j] for i, j in off])
    w = 0.6 * w / jnp.abs(w).max()
    b = 0.4 * jnp.tanh(jnp.diag(C))
    return off, w, b


pairs, weights, biases = ising_from_matrix(C)
nodes = [SpinNode() for _ in range(N)]
edges = [(nodes[i], nodes[j]) for i, j in pairs]
model = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))

free_blocks = [Block([n]) for n in nodes]  # fully connected graph -> one node per block
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
schedule = SamplingSchedule(n_warmup=200, n_samples=500, steps_per_sample=2)

n_chains = 64
init = hinton_init(k_init, model, free_blocks, (n_chains,))
chain_keys = jax.random.split(k_run, n_chains)


def run_chain(k, init_state):
    return sample_states(k, program, schedule, init_state, [], [Block(nodes)])[0]


samples = jax.vmap(run_chain)(chain_keys, init)  # (chains, samples, N) bool
spins = (2 * samples.astype(jnp.float32) - 1).reshape(-1, N)  # -> {-1, +1}
M = spins.T @ spins / spins.shape[0]  # correlation matrix <s_i s_j>

# Exact reference: enumerate all 2^N states, E(s) = -(sum b_i s_i + sum J_ij s_i s_j)
states = jnp.array(list(itertools.product([-1.0, 1.0], repeat=N)))
pair_idx = jnp.array(pairs)
energy = -(
    states @ biases + (states[:, pair_idx[:, 0]] * states[:, pair_idx[:, 1]]) @ weights
)
p = jax.nn.softmax(-energy)
M_exact = (states * p[:, None]).T @ states

err_thrml = float(jnp.abs(M - M_exact).max())
assert err_thrml < 0.05, err_thrml
print("Stage 2  thrml      M = <s_i s_j> from", spins.shape[0], "samples")
print(np.round(np.asarray(M), 3))
print(f"         max |M - exact| = {err_thrml:.3f} (Monte Carlo error)\n")


# ---------------------------------------------------------------------------
# Stage 3: PennyLane -- each row of M is angle-encoded into a 4-qubit circuit
# ---------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=N)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(features, weights):
    qml.AngleEmbedding(features * jnp.pi / 2, wires=range(N), rotation="Y")
    for k in range(N):
        qml.RY(weights[k], wires=k)
    for k in range(N - 1):
        qml.CNOT(wires=[k, k + 1])
    return [qml.expval(qml.PauliZ(k)) for k in range(N)]


q_weights = jnp.array([0.1, -0.2, 0.3, -0.1])
Q = jnp.stack(circuit(M, q_weights), axis=-1)  # broadcast over the 4 rows -> (4, 4)

# Exact reference: after RY on |0> and a CNOT chain, <Z_k> = prod_{i<=k} cos(theta_i)
theta = M * jnp.pi / 2 + q_weights
Q_exact = jnp.cumprod(jnp.cos(theta), axis=1)

err_pl = float(jnp.abs(Q - Q_exact).max())
assert err_pl < 1e-5, err_pl
print("Stage 3  PennyLane  Q = <Z_k> per row of M")
print(np.round(np.asarray(Q), 3))
print(f"         max |Q - exact| = {err_pl:.1e}\n")


# ---------------------------------------------------------------------------
# Stage 4: Q.ANT -- photonic matmul through the bridge (works without an NPU)
# ---------------------------------------------------------------------------
# The bridge picks the backend: real NPU > cpu-backend simulator > NumPy
# emulation. Force one with QANT_BACKEND=npu|sim|emulate.
backend = detect_backend()

# 4a. Train the layer weights in JAX, hardware-aware: bfloat16 rounding plus
#     output noise, so the weights still work on analog hardware later.
target = jnp.array(
    [[1.0, -1.0], [0.5, 0.0], [-0.5, 1.0], [0.0, 0.5]]
)  # what Y should be
W = jax.random.normal(k_w, (2, N)) * 0.5


def train_loss(w, k):
    y = hw_aware_linear(Q, w, key=k, noise_std=0.04)
    return jnp.mean((y - target) ** 2)


@jax.jit
def train_step(w, k):
    loss, g = jax.value_and_grad(train_loss)(w, k)
    return w - 0.3 * g, loss


k_train = jax.random.key(1)
for step in range(300):
    k_train, k_step = jax.random.split(k_train)
    W, loss = train_step(W, k_step)

# 4b. Deploy: weights uploaded once to the backend, features sent per call.
layer = QantLinear(W, backend=backend)
Y = layer(Q)

Y_ref = Q @ W.T
err_qant = float(jnp.abs(Y - Y_ref).max())
assert err_qant < 0.05, err_qant  # bfloat16 has ~3 significant digits
deploy_loss = float(jnp.mean((Y - target) ** 2))
print("Stage 4  Q.ANT      Y = Q @ W.T")
print("        ", layer.report().replace("\n", "\n         "))
print(np.round(np.asarray(Y), 3))
print(f"         max |Y - float32| = {err_qant:.4f} (bfloat16 rounding)")
print(f"         loss: trained {float(loss):.4f} -> deployed {deploy_loss:.4f}\n")


# ---------------------------------------------------------------------------
# Bonus: gradients flow through PennyLane and the Q.ANT layer together
# ---------------------------------------------------------------------------
def loss_fn(qw):
    q = jnp.stack(circuit(M, qw), axis=-1)
    return jnp.sum(layer(q) ** 2)


g_qw = jax.grad(loss_fn)(q_weights)
print("Bonus    d loss / d quantum weights =", np.round(np.asarray(g_qw), 4))
print("All four stages match their references.")

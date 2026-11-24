-- Functions exported to C with stable, unmangled symbol names.
-- Scalar types map directly to C types: no boxing, no reference counting.

@[export pylean_add]
def add (a b : UInt64) : UInt64 := a + b

def fibAux : Nat → UInt64 → UInt64 → UInt64
  | 0,     a, _ => a
  | n + 1, a, b => fibAux n b (a + b)

@[export pylean_fib]
def fib (n : UInt64) : UInt64 := fibAux n.toNat 0 1

@[export pylean_hypot]
def hypot (x y : Float) : Float := Float.sqrt (x * x + y * y)

@[export pylean_is_prime]
def isPrime (n : UInt64) : Bool := Id.run do
  if n < 2 then return false
  let mut i : UInt64 := 2
  while i * i ≤ n do
    if n % i == 0 then return false
    i := i + 1
  return true

-- Strings are heap objects (lean_object*).
-- The argument is "owned": Lean takes it and frees it, so the caller must not.
@[export pylean_greet]
def greet (name : String) : String :=
  s!"Hello from Lean 4, {name}! (length {name.length})"
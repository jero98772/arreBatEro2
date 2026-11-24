#!/usr/bin/env python3
"""
Call Lean 4 functions from Python through the C foreign function interface.

Pure FFI: this script only uses `ctypes` to load already-compiled shared
libraries and call their C symbols. It does not start any external process.

Step 1 (once, in your terminal):
    cd lean_project
    lake build
  This compiles PyLean.lean into a shared library (.so / .dylib / .dll).

Step 2:
    python3 python_lean_ffi.py [path/to/libPyLean.so] [path/to/libleanshared.so]
  With no arguments it looks in ./lean_project/.lake/build and in ~/.elan.
  You can also set LEAN_SYSROOT to your Lean toolchain folder.

Type mapping (Lean -> C -> ctypes)
  UInt64 -> uint64_t     -> c_uint64
  Float  -> double       -> c_double
  Bool   -> uint8_t      -> c_uint8
  String -> lean_object* -> c_void_p   (heap object, reference counted)
"""

import ctypes
import glob
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE / "lean_project"
MODULE = "PyLean"
PACKAGE = "pylean"
LIB_EXTS = ("so", "dylib", "dll")


# --------------------------------------------------------------------------
# Locating the libraries (filesystem only)
# --------------------------------------------------------------------------
def find_user_lib():
    for ext in LIB_EXTS:
        hits = glob.glob(
            str(PROJECT_DIR / ".lake" / "build" / "**" / f"*{MODULE}*.{ext}"),
            recursive=True,
        )
        if hits:
            return hits[0]
    sys.exit(
        "Shared library not found. Run `lake build` inside lean_project first,\n"
        "or pass its path as the first argument."
    )


def find_lean_runtime():
    roots = []
    if os.environ.get("LEAN_SYSROOT"):
        roots.append(Path(os.environ["LEAN_SYSROOT"]))

    # Prefer the toolchain named in the project's lean-toolchain file,
    # e.g. "leanprover/lean4:v4.24.0" -> ~/.elan/toolchains/leanprover--lean4---v4.24.0
    elan = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan")) / "toolchains"
    tc_file = PROJECT_DIR / "lean-toolchain"
    if tc_file.exists():
        name = tc_file.read_text().strip().replace("/", "--").replace(":", "---")
        roots.append(elan / name)
    roots += sorted(elan.glob("*"), reverse=True)

    for root in roots:
        for sub in ("lib/lean", "lib", "bin"):
            for ext in LIB_EXTS:
                hits = glob.glob(str(root / sub / f"libleanshared.{ext}"))
                if hits:
                    return hits[0]
    sys.exit(
        "libleanshared not found. Set LEAN_SYSROOT to your Lean toolchain folder,\n"
        "or pass the library path as the second argument."
    )


# --------------------------------------------------------------------------
# The FFI binding
# --------------------------------------------------------------------------
class LeanFFI:
    # Object layout from lean.h (64-bit):
    #   lean_object: int32 m_rc | uint16 m_cs_sz | uint8 m_other | uint8 m_tag  (8 bytes)
    #   lean_string_object: header + m_size, m_capacity, m_length (size_t each) + UTF-8 bytes
    TAG_OFFSET = 7
    STRING_DATA_OFFSET = 8 + 3 * ctypes.sizeof(ctypes.c_size_t)

    def __init__(self, lib_path, runtime_path):
        if platform.system() == "Windows":
            os.add_dll_directory(str(Path(runtime_path).parent))
        # Load the runtime globally first: our library resolves its symbols from it.
        self.rt = ctypes.CDLL(runtime_path, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._declare_signatures()
        self._initialize()

    def _declare_signatures(self):
        rt, lib = self.rt, self.lib
        c_u64, c_f64, c_u8, c_ptr = (
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.c_uint8,
            ctypes.c_void_p,
        )

        # Lean runtime functions (exported, not `static inline`)
        rt.lean_initialize_runtime_module.argtypes = []
        rt.lean_initialize_runtime_module.restype = None
        rt.lean_io_mark_end_initialization.argtypes = []
        rt.lean_io_mark_end_initialization.restype = None
        rt.lean_mk_string.argtypes = [ctypes.c_char_p]
        rt.lean_mk_string.restype = c_ptr
        rt.lean_dec_ref_cold.argtypes = [c_ptr]
        rt.lean_dec_ref_cold.restype = None

        # Our exported Lean functions
        lib.pylean_add.argtypes, lib.pylean_add.restype = [c_u64, c_u64], c_u64
        lib.pylean_fib.argtypes, lib.pylean_fib.restype = [c_u64], c_u64
        lib.pylean_hypot.argtypes, lib.pylean_hypot.restype = [c_f64, c_f64], c_f64
        lib.pylean_is_prime.argtypes, lib.pylean_is_prime.restype = [c_u64], c_u8
        lib.pylean_greet.argtypes, lib.pylean_greet.restype = [c_ptr], c_ptr

    def _initialize(self):
        # Same steps Lean's generated main() performs.
        self.rt.lean_initialize_runtime_module()

        # Module initializer name differs between Lean/Lake versions.
        init = None
        for name in (f"initialize_{MODULE}", f"initialize_{PACKAGE}_{MODULE}"):
            try:
                init = getattr(self.lib, name)
                break
            except AttributeError:
                continue
        if init is None:
            raise RuntimeError(
                f"No initializer symbol found (tried initialize_{MODULE})."
            )

        init.argtypes = [ctypes.c_uint8, ctypes.c_void_p]
        init.restype = ctypes.c_void_p
        world = ctypes.c_void_p(1)  # lean_io_mk_world() == lean_box(0) == (void*)1
        res = init(1, world)  # builtin = 1
        # IO result is a heap object: tag 0 = ok, 1 = error
        if res and not (res & 1):
            if ctypes.c_uint8.from_address(res + self.TAG_OFFSET).value != 0:
                raise RuntimeError("Lean module initialization failed")
        self.rt.lean_io_mark_end_initialization()

    # Reference counting (mirrors the inline lean_dec_ref from lean.h)
    def dec_ref(self, obj):
        if not obj or obj & 1:  # null, or a boxed scalar
            return
        rc = ctypes.c_int32.from_address(obj)
        if rc.value > 1:
            rc.value -= 1
        elif rc.value != 0:  # 0 = persistent object, never freed
            self.rt.lean_dec_ref_cold(obj)

    def _to_lean_string(self, s):
        return self.rt.lean_mk_string(s.encode("utf-8"))

    def _from_lean_string(self, obj):
        return ctypes.string_at(obj + self.STRING_DATA_OFFSET).decode("utf-8")

    # Pythonic wrappers
    def add(self, a, b):
        return self.lib.pylean_add(a, b)

    def fib(self, n):
        return self.lib.pylean_fib(n)

    def hypot(self, x, y):
        return self.lib.pylean_hypot(x, y)

    def is_prime(self, n):
        return bool(self.lib.pylean_is_prime(n))

    def greet(self, name):
        arg = self._to_lean_string(name)
        res = self.lib.pylean_greet(arg)  # Lean consumes arg; we own the result
        try:
            return self._from_lean_string(res)
        finally:
            self.dec_ref(res)  # do NOT dec_ref(arg): Lean already freed it


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def main():
    lib_path = sys.argv[1] if len(sys.argv) > 1 else find_user_lib()
    runtime_path = sys.argv[2] if len(sys.argv) > 2 else find_lean_runtime()
    print(f"Lean library : {lib_path}\nLean runtime : {runtime_path}\n")

    lean = LeanFFI(lib_path, runtime_path)

    print("add(40, 2)        =", lean.add(40, 2))
    print("fib(90)           =", lean.fib(90))
    print("hypot(3.0, 4.0)   =", lean.hypot(3.0, 4.0))
    print("primes below 30   =", [n for n in range(30) if lean.is_prime(n)])
    print("greet('Python')   =", lean.greet("Python"))
    print("greet('Medellín') =", lean.greet("Medellín"))


if __name__ == "__main__":
    main()

import Lake
open Lake DSL

package pylean

@[default_target]
lean_lib PyLean where
  -- build a shared library (.so / .dylib / .dll) instead of only .olean files
  defaultFacets := #[LeanLib.sharedFacet]
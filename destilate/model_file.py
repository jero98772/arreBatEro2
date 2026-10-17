"""Adapter so inference.py / destilate.py's `module:Class` + flat-JSON-kwargs
loading convention can build HybridTransformerLM (hybrid/step3_hybrid_autoregrsive.py),
whose real constructor takes a nested cfg/smap rather than flat kwargs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid.step3_hybrid_autoregrsive import CFG, HybridTransformerLM

# hybrid.pt's own arch / substrate_map (read from the checkpoint's metadata).
_ARCH = {
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "n_qubits": 8,
    "sim_backend": "torch",
    "aft_ksize": 4,
}
_SMAP = {
    "AR_EMB": "classical",
    "AR_ATT1": "photonic",
    "AR_ATT2": "photonic",
    "AR_MLP1": "photonic",
    "AR_MLP2": "photonic",
    "AR_NORM1": "classical",
    "AR_NORM2": "classical",
    "AR_HEAD": "photonic",
}


class HybridStudent(HybridTransformerLM):
    def __init__(self, vocab_size, max_len=512, **overrides):
        cfg = {**CFG, **_ARCH, "seq_len": max_len - 1}
        cfg.update(overrides)
        super().__init__(vocab_size, cfg, dict(_SMAP))


YourModelClass = HybridStudent

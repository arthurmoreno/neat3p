"""Shared loader for the committed torch-parity fixtures (no torch needed)."""

from __future__ import annotations

import json

import make_fixtures as mf
import numpy as np

from neat3p.genome import DefaultGenome

_META = None
_ARRAYS = None
_CONFIGS: dict = {}


def _load():
    global _META, _ARRAYS
    if _META is None:
        with open(mf.CASES_JSON) as f:
            _META = json.load(f)
        _ARRAYS = dict(np.load(mf.ARRAYS_NPZ))
    return _META, _ARRAYS


def case_names() -> list[str]:
    return [c["name"] for c in _load()[0]["cases"]]


def config_for(name: str):
    if name not in _CONFIGS:
        _CONFIGS[name] = mf.load_config(name)
    return _CONFIGS[name]


def load_case(case_name: str):
    meta, arrays = _load()
    case = next(c for c in meta["cases"] if c["name"] == case_name)
    genome = DefaultGenome.model_validate_json(json.dumps(case["genome"]))
    config = config_for(case["config"])
    prefix = case_name + "/"
    own = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
    return genome, config, own, meta["reset_at"]


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a-b| over finite entries; NaN/inf positions must match exactly."""
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN positions differ"
    assert np.array_equal(np.isinf(a), np.isinf(b)), "inf positions differ"
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0


ATOL64 = 1e-12
NEAR_TIE = 1e-9


def run_sequence(step, reset, inputs: np.ndarray, reset_at) -> np.ndarray:
    """Feed ``inputs`` one tick at a time, calling ``reset()`` before each tick in ``reset_at``."""
    outs = []
    for t in range(inputs.shape[0]):
        if t in reset_at:
            reset()
        outs.append(np.array(step(inputs[t]), dtype=np.float64).reshape(-1))
    return np.stack(outs)


def assert_actions_match(got: np.ndarray, want: np.ndarray) -> None:
    """argmax must agree on every finite row whose top-2 gap is at least NEAR_TIE."""
    finite = np.isfinite(want).all(axis=1)
    if want.shape[1] > 1:
        top2 = np.sort(want, axis=1)[:, -2:]
        decisive = finite & ((top2[:, 1] - top2[:, 0]) >= NEAR_TIE)
    else:
        decisive = finite
    assert np.array_equal(np.argmax(got, axis=1)[decisive], np.argmax(want, axis=1)[decisive])

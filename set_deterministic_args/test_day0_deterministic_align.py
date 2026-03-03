import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from day0_deterministic_align import (
    CompareSummary,
    _load_rollout,
    _save_rollout,
    build_sampling_params,
    compare_logprobs,
    save_manifest,
)


def _make_args(**kwargs):
    defaults = dict(
        host="localhost",
        port=30000,
        model_path="Qwen/Qwen3-8B",
        attn_implementation="sdpa",
        prompt="hello",
        max_new_tokens=8,
        device="cpu",
        dtype="bf16",
        seed=42,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        tolerance=1e-6,
        strict_bitwise=True,
        use_batch_invariant=False,
        save_rollout=None,
        load_rollout=None,
        output_json=None,
        save_detail=None,
        manifest_json=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_build_sampling_params_has_all_fields():
    args = _make_args()
    params = build_sampling_params(args)
    assert set(params.keys()) == {
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "max_new_tokens",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "sampling_seed",
    }
    assert params["temperature"] == 1.0
    assert params["top_k"] == 1
    assert params["sampling_seed"] == 42


def test_compare_logprobs_bitwise_and_numeric():
    sglang = [0.1, 0.2, 0.3]
    hf = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    details, summary = compare_logprobs(sglang, hf, tolerance=1e-7)
    assert isinstance(summary, CompareSummary)
    assert summary.total == 3
    assert summary.numeric_all_match
    assert summary.bitwise_all_match
    assert all(d["numeric_match"] for d in details)
    assert all(d["bitwise_match"] for d in details)


def test_compare_logprobs_numeric_match_but_not_bitwise():
    # nextafter creates a one-ULP difference.
    base = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    shifted = torch.nextafter(base, torch.tensor(float("inf"), dtype=torch.float32))
    sglang = [float(x) for x in shifted.tolist()]
    details, summary = compare_logprobs(sglang, base, tolerance=1e-3)
    assert summary.numeric_all_match
    assert not summary.bitwise_all_match
    assert summary.first_bitwise_mismatch is not None
    assert any((d["numeric_match"] and (not d["bitwise_match"])) for d in details)


def test_rollout_save_and_load_roundtrip(tmp_path: Path):
    rollout_path = tmp_path / "rollout.json"
    payload = {
        "tokens": [1, 2, 3, 4],
        "rollout_logprobs": [-1.0, -2.0],
        "prompt_len": 2,
        "gen_token_ids": [3, 4],
        "gen_token_texts": ["a", "b"],
    }
    _save_rollout(rollout_path, payload)
    loaded = _load_rollout(rollout_path)
    assert loaded == payload


def test_save_manifest_contains_required_blocks(tmp_path: Path):
    args = _make_args()
    manifest = tmp_path / "manifest.json"
    save_manifest(manifest, args, prompt_ids=[101, 102, 103])
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "timestamp" in data
    assert "args" in data
    assert "runtime" in data
    assert "env" in data
    assert data["prompt_ids"] == [101, 102, 103]

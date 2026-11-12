"""The Qwen slice: every valid slice edit stands for a real low-rank edit that reproduces it exactly.
Skipped when the slices have not been built (they need the Qwen3.5-0.8B weights on disk)."""
import json
import os

import numpy as np
import pytest

from weightgym import qwen_slice as qs
from weightgym.edits import EditError, apply_edit

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(qs.SLICES, "layer22.npz")), reason="Qwen slices not built")


@pytest.fixture(scope="module")
def item():
    return qs.SliceItem(22)


def test_slice_shape_and_prompt(item):
    assert item.net.Ws[0].shape == (6, 3584) and len(item.shown) == 64
    assert np.allclose(item.U @ item.G, np.eye(6), atol=1e-8)
    p = item.prompt(True)
    assert len(p) < 40_000 and '"</think>"' in p and "layer 22" in p
    assert abs(item.net.Ws[0][:, item.shown].std() - 1.0) < 1e-6


def _edits(item):
    a, b = int(item.shown[0]), int(item.shown[5])
    return [
        [{"op": "scale", "tensor": "W1", "axis": "col", "index": a, "factor": 2.0}],
        [{"op": "scale", "tensor": "W1", "axis": "row", "index": 0, "factor": 0.0}],
        [{"op": "swap", "tensor": "W1", "axis": "col", "i": a, "j": b}],
        [{"op": "swap", "tensor": "W1", "axis": "row", "i": 0, "j": 2}],
        [{"op": "set", "tensor": "W1", "index": [2, a], "value": 1.5}],
        [{"op": "add", "tensor": "W1", "axis": "col", "index": b, "vector": [0.5, -0.5, 1, 0, 0, 2]}],
        [{"op": "set", "tensor": "W1", "index": [0, a], "value": -3.0}, {"op": "scale", "tensor": "W1", "axis": "row", "index": 0, "factor": 0.5},
         {"op": "swap", "tensor": "W1", "axis": "col", "i": a, "j": b}],
    ]


def test_real_edit_reproduces_the_slice_edit(item):
    for ops in _edits(item):
        ops = item.validate(ops)
        want = apply_edit(item.net, ops).Ws[0]
        delta = item.real_delta(ops)
        assert np.allclose(item.U @ (item.W + delta), want, atol=1e-7), ops
        B, A = item.factors(ops)
        assert B.shape[1] <= 8 and np.allclose(B @ A, delta, atol=1e-7), ops


def test_column_scale_is_the_published_kind_of_edit(item):
    j = int(item.shown[0])
    d = item.real_delta(item.validate([{"op": "scale", "tensor": "W1", "axis": "col", "index": j, "factor": 2.0}]))
    assert np.allclose(d[:, j], item.W[:, j]) and np.count_nonzero(np.abs(d).sum(0)) == 1


def test_invalid_answers(item):
    hidden = next(j for j in range(3584) if j not in set(item.shown.tolist()))
    for bad in ([{"op": "set", "tensor": "b1", "index": 0, "value": 1.0}],
                [{"op": "scale", "tensor": "W1", "axis": "col", "index": hidden, "factor": 2.0}],
                [{"op": "set", "tensor": "W2", "index": [0, 0], "value": 1.0}]):
        with pytest.raises(EditError):
            item.validate(bad)
    assert item.score("[]")["valid"] and item.score("[]")["noop"] and not item.score("nonsense")["valid"]


def test_random_edits_are_valid_and_deterministic(item):
    a = [qs.random_valid_edit(item, np.random.default_rng([item.seed, 0])) for _ in range(3)]
    b = [qs.random_valid_edit(item, np.random.default_rng([item.seed, 0])) for _ in range(3)]
    assert a == b and all(item.validate(x) for x in a)
    assert qs.edit_id(22, a[0]) == qs.edit_id(22, json.loads(json.dumps(a[0])))

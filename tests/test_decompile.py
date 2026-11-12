import numpy as np
import pytest

from weightgym.decompile import (
    SUCCESS_THRESHOLD, DecompileItem, ExprError, agreement, compile_expr, extract_expr, reference_expr,
)
from weightgym.env import GymConfig, make_episode
from weightgym.nets import FAMILIES, sample_task


@pytest.mark.parametrize("family", list(FAMILIES))
def test_reference_expression_is_exact(family):
    rng = np.random.default_rng(5)
    for size in (0, 1, 2):
        for _ in range(15):
            task, net = sample_task(family, rng, size)
            assert agreement(task, net, reference_expr(task)) == 1.0, reference_expr(task)


def test_items_from_episodes_score_their_reference():
    cfg = GymConfig(families=tuple(FAMILIES))
    for seed in range(60):
        item = DecompileItem(make_episode(seed, cfg))
        res = item.score(item.target_text())
        assert res["valid"] and res["success"] and res["recovery"] >= SUCCESS_THRESHOLD
        assert "Target:" not in item.prompt() and item.ep.task.spec not in item.prompt()  # the spec is hidden
        assert item.prompt(direct=True) != item.prompt() and item.net.equals(item.ep.clean)
        assert item.score("nonsense(")["reward"] == -1.0
        wrong = item.score("x[0] > 99")  # valid expression, wrong function
        assert wrong["valid"] and 0.0 <= wrong["reward"] <= 1.0


def test_equivalent_expression_counts():
    task, net = None, None
    rng = np.random.default_rng(1)
    while task is None or task.params["mode"] != "max2":
        task, net = sample_task("minmax", rng, 0)
    a, b = task.params["vars"]
    assert agreement(task, net, f"x[{a}] if x[{a}] > x[{b}] else x[{b}]") == 1.0
    assert agreement(task, net, f"x[{b}] + max(x[{a}] - x[{b}], 0)") == 1.0


BAD = ["__import__('os')", "x.__class__", "[i for i in x]", "lambda: 1", "x[0] ** 9", "open('f')", "x * 99999",
       "9999999", "'a'", "x[0]; x[1]", "", "abs(x[0], key=1)", "x[0] if"]


@pytest.mark.parametrize("text", BAD)
def test_evaluator_rejects(text):
    task, net = sample_task("threshold", np.random.default_rng(0), 0)
    with pytest.raises(ExprError):
        agreement(task, net, text)


def test_extract_expr_takes_last_line():
    assert extract_expr("thinking...\n```python\nx[0] >= 1\n```") == "x[0] >= 1"
    assert extract_expr("`max(x[0], x[1])`") == "max(x[0], x[1])"
    assert compile_expr("argmax([x[0] + x[1], x[2]])")((1, 2, 5)) == 1

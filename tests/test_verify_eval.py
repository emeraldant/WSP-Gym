from weightgym.verify_eval import _distinct, fixed_within


def _recs():
    # network 1: fixed by its 3rd answer; network 2: never fixed; network 3: fixed by its 1st answer
    rows = []
    for seed, wins in ((1, {2}), (2, set()), (3, {0})):
        for j in range(4):
            rows.append({"seed": seed, "sample": j, "success": j in wins, "ops": [{"op": "set", "v": j % 2}], "answer": "x"})
    return rows


def test_fixed_within_counts_a_network_once_and_grows_with_k():
    recs = _recs()
    assert fixed_within(recs, 1)[0] == 1 / 3
    assert fixed_within(recs, 2)[0] == 1 / 3
    assert fixed_within(recs, 4)[0] == 2 / 3
    assert fixed_within(recs, 4)[3] == 3


def test_distinct_answers_per_network():
    assert _distinct(_recs(), 4) == [2, 2, 2]
    assert _distinct(_recs(), 1) == [1, 1, 1]

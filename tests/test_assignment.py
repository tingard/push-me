from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from push_me.goals import resolve_assignment

_cost_matrices = st.integers(min_value=1, max_value=5).flatmap(
    lambda k: arrays(
        dtype=float,
        shape=(k, k),
        elements=st.floats(
            min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
    )
)


def _brute_force_min_cost(cost: np.ndarray) -> float:
    k = cost.shape[0]
    return min(
        sum(cost[i, perm[i]] for i in range(k))
        for perm in itertools.permutations(range(k))
    )


@given(_cost_matrices)
def test_free_assignment_matches_brute_force_minimum(cost):
    _assignment, errors = resolve_assignment(cost, mode="free")
    assert errors.sum() == pytest.approx(_brute_force_min_cost(cost), abs=1e-6)


@given(_cost_matrices)
def test_free_assignment_is_a_valid_permutation(cost):
    k = cost.shape[0]
    assignment, errors = resolve_assignment(cost, mode="free")
    assert sorted(assignment.tolist()) == list(range(k))
    assert errors.tolist() == pytest.approx([cost[i, assignment[i]] for i in range(k)])


@given(_cost_matrices)
def test_fixed_assignment_is_the_identity(cost):
    k = cost.shape[0]
    assignment, errors = resolve_assignment(cost, mode="fixed")
    assert assignment.tolist() == list(range(k))
    assert errors.tolist() == pytest.approx([cost[i, i] for i in range(k)])


@given(_cost_matrices)
def test_free_assignment_total_cost_is_invariant_to_row_permutation(cost):
    k = cost.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(k)
    shuffled = cost[perm]

    _assignment, errors = resolve_assignment(cost, mode="free")
    _shuffled_assignment, shuffled_errors = resolve_assignment(shuffled, mode="free")

    assert shuffled_errors.sum() == pytest.approx(errors.sum(), abs=1e-6)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_assignment(np.zeros((2, 2)), mode="bogus")

import pytest

from moe_eviction.cache import HardLRUCache


def reference_lru(
    requests: tuple[tuple[int, ...], ...],
    capacity: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
    resident: list[int] = []
    results = []
    for request in requests:
        before = tuple(resident)
        before_set = set(before)
        hits = tuple(expert for expert in request if expert in before_set)
        misses = tuple(expert for expert in request if expert not in before_set)
        for expert in request:
            if expert in resident:
                resident.remove(expert)
            resident.append(expert)
            while len(resident) > capacity:
                resident.pop(0)
        results.append((hits, misses, tuple(resident)))
    return tuple(results)


def test_hits_are_evaluated_against_pre_step_snapshot() -> None:
    cache = HardLRUCache(capacity=3, initial=(1, 2, 3))

    result = cache.access((2, 4))

    assert result.resident_before == (1, 2, 3)
    assert result.hits == (2,)
    assert result.misses == (4,)
    assert result.resident_after == (3, 2, 4)
    assert result.hit_count + result.miss_count == len(result.requested)


def test_capacity_smaller_than_top_k_is_deterministic() -> None:
    cache = HardLRUCache(capacity=2)

    result = cache.access((4, 5, 6))

    assert result.misses == (4, 5, 6)
    assert result.resident_after == (5, 6)


def test_capacity_equal_to_top_k_keeps_request_order() -> None:
    cache = HardLRUCache(capacity=3, initial=(7, 8, 9))

    result = cache.access((1, 2, 3))

    assert result.hits == ()
    assert result.misses == (1, 2, 3)
    assert result.resident_after == (1, 2, 3)


def test_capacity_larger_than_top_k_retains_unrecent_residents() -> None:
    cache = HardLRUCache(capacity=4, initial=(0, 1))

    result = cache.access((2, 3))

    assert result.resident_after == (0, 1, 2, 3)


def test_hit_moves_expert_to_mru_before_later_insertion() -> None:
    cache = HardLRUCache(capacity=3, initial=(1, 2, 3))

    result = cache.access((1, 4))

    assert result.resident_after == (3, 1, 4)


def test_all_hit_step_reports_exact_hit_order() -> None:
    cache = HardLRUCache(capacity=4, initial=(0, 1, 2, 3))

    result = cache.access((2, 0))

    assert result.hits == (2, 0)
    assert result.misses == ()
    assert result.resident_after == (1, 3, 2, 0)


def test_mixed_step_reports_exact_miss_identity() -> None:
    cache = HardLRUCache(capacity=3, initial=(0, 1, 2))

    result = cache.access((3, 1, 4))

    assert result.hits == (1,)
    assert result.misses == (3, 4)
    assert result.resident_after == (3, 1, 4)


def test_duplicate_request_is_rejected() -> None:
    cache = HardLRUCache(capacity=2)
    with pytest.raises(ValueError, match="unique"):
        cache.access((1, 1))


def test_empty_request_is_rejected() -> None:
    cache = HardLRUCache(capacity=2)
    with pytest.raises(ValueError, match="empty"):
        cache.access(())


def test_negative_request_is_rejected() -> None:
    cache = HardLRUCache(capacity=2)
    with pytest.raises(ValueError, match="non-negative"):
        cache.access((-1,))


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        HardLRUCache(capacity=0)


def test_invalid_initial_cache_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        HardLRUCache(capacity=2, initial=(1, 1))
    with pytest.raises(ValueError, match="exceed"):
        HardLRUCache(capacity=1, initial=(1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        HardLRUCache(capacity=2, initial=(-1,))


def test_exhaustive_small_traces_match_reference() -> None:
    requests = (
        ((0,), (1,), (0, 1), (2,)),
        ((0, 1), (1, 2), (2, 0)),
        ((2, 1), (0,), (1, 2), (0, 2)),
    )
    for capacity in (1, 2, 3):
        cache = HardLRUCache(capacity=capacity)
        expected = reference_lru(requests[capacity - 1], capacity)
        for request, (hits, misses, resident_after) in zip(
            requests[capacity - 1],
            expected,
            strict=True,
        ):
            result = cache.access(request)
            assert result.hits == hits
            assert result.misses == misses
            assert result.resident_after == resident_after

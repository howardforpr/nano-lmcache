import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nano_lmcache.storage.policy import LRU, MRU

BENCH_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_prefix_cache.py"
)


def load_bench() -> ModuleType:
    """Load the benchmark script by path; benchmarks/ is not a package."""
    spec = importlib.util.spec_from_file_location("bench_prefix_cache", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_returns_one_result_per_scenario():
    bench = load_bench()

    results = bench.run_benchmark(
        chunk_counts=[4, 8],
        capacities=[None, 4],
        repeats=2,
    )

    assert len(results) == 4
    assert {(r.chunk_count, r.capacity) for r in results} == {
        (4, None),
        (4, 4),
        (8, None),
        (8, 4),
    }


def test_benchmark_results_are_positive_timings():
    bench = load_bench()

    (result,) = bench.run_benchmark(
        chunk_counts=[4], capacities=[None], repeats=2
    )

    assert result.store_seconds > 0
    assert result.lookup_seconds > 0


def test_benchmark_reports_expected_hits():
    bench = load_bench()

    unbounded, bounded = bench.run_benchmark(
        chunk_counts=[8], capacities=[None, 4], repeats=1
    )

    # Unbounded keeps the whole prompt; capacity 4 keeps only a 4-chunk prefix.
    assert unbounded.matched_chunks == 8
    assert bounded.matched_chunks == 4


def test_format_table_has_a_row_per_result():
    bench = load_bench()
    results = bench.run_benchmark(
        chunk_counts=[4], capacities=[None, 2], repeats=1
    )

    table = bench.format_table(results)

    lines = table.splitlines()
    assert len(lines) == 2 + len(results)
    assert "chunks" in lines[0]
    assert "unbounded" in table


def test_churn_scenario_keeps_only_capacity_many_entries():
    bench = load_bench()

    result = bench.run_churn(
        prompts=20, chunks_per_prompt=4, capacity=8, repeats=1
    )

    assert result.seconds > 0
    # 80 chunks were written, so most must have been evicted.
    assert 0 < result.entries <= 8


def test_churn_without_capacity_keeps_everything():
    bench = load_bench()

    result = bench.run_churn(
        prompts=5, chunks_per_prompt=4, capacity=None, repeats=1
    )

    assert result.entries == 20


def test_deep_chain_churn_fills_large_capacity_and_forces_eviction():
    bench = load_bench()

    result = bench.run_deep_chain_churn(repeats=1)

    assert result.prompts == 64
    assert result.chunks_per_prompt == 1024
    assert result.capacity == 8192
    assert result.prompts * result.chunks_per_prompt > result.capacity
    assert result.entries == result.capacity


def test_cli_accepts_deep_chain_churn_parameters():
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH_PATH),
            "--chunks",
            "4",
            "--capacity",
            "2",
            "--repeats",
            "1",
            "--churn-prompts",
            "3",
            "--churn-chunks",
            "5",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "3 prompts x 5 chunks, capacity unbounded:" in result.stdout
    assert "3 prompts x 5 chunks, capacity 2:" in result.stdout
    assert "2 of 15 chunks kept" in result.stdout
    assert "Deep-chain eviction churn (large capacity):" in result.stdout
    assert "64 prompts x 1024 chunks, capacity 8192:" in result.stdout
    assert "8192 of 65536 chunks kept" in result.stdout
    assert "Shared system prompts" not in result.stdout


@pytest.mark.parametrize("policy", [LRU(), MRU()])
def test_policy_benchmark_counts_actual_prefix_hits(policy):
    bench = load_bench()
    a = [0] * bench.CHUNK_SIZE + [1] * bench.CHUNK_SIZE
    b = [0] * bench.CHUNK_SIZE + [2] * bench.CHUNK_SIZE

    metrics = bench.run_policy_workload([a, b, a], policy=policy, capacity=4)

    assert metrics.hit_chunks == 3  # Cold A, shared first chunk, fully cached A.
    assert metrics.total_chunks == 6
    assert metrics.hit_rate == 0.5


def test_shared_prompt_traces_have_diverse_tails_and_repeatable_order():
    bench = load_bench()
    for pattern in ("hot", "scan"):
        trace = bench.shared_prompt_trace(pattern, requests=100)
        assert len(trace) == 100
        assert trace == bench.shared_prompt_trace(pattern, requests=100)
        assert {len(request) for request in trace} == {12 * bench.CHUNK_SIZE}
        assert len({tuple(request[:4 * bench.CHUNK_SIZE]) for request in trace}) == 2
        assert len({tuple(request) for request in trace}) > 8


def test_policy_benchmark_cli_reports_both_policies_and_locality_patterns():
    result = subprocess.run(
        [
            sys.executable, str(BENCH_PATH), "--policy-only", "--policy-requests",
            "20", "--capacity", "32",
        ],
        capture_output=True, text=True, check=True,
    )

    assert "LRU" in result.stdout and "MRU" in result.stdout
    assert "hot" in result.stdout and "scan" in result.stdout
    assert "hit rate" in result.stdout

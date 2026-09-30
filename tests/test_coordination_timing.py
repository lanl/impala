"""Coordinator instrumentation preserves heterogeneous requests and retirement."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from gpu_batched_chains import serve


class Connection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    def poll(self, timeout):
        return True

    def recv(self):
        return next(self.messages)

    def send(self, value):
        self.sent.append(value)


class Model:
    def eval(self, params, pool):
        return (params["x"] * 2)[:, None]


@pytest.mark.parametrize("timing", [False, True])
def test_timing_preserves_requests_and_retiring_worker(timing):
    def request(values):
        params = {"x": np.array(values)}
        return (
            ("evaluate_timed", (params, 0.0))
            if timing
            else ("evaluate", params)
        )

    connections = {
        42: Connection([request([1, 2]), ("sampled", 1.0)]),
        123: Connection([request([3]), request([4, 5, 6]), ("sampled", 2.0)]),
    }
    result = serve(connections, Model(), "sampled", timing=timing)
    np.testing.assert_array_equal(connections[42].sent[0][1], [[2], [4]])
    np.testing.assert_array_equal(connections[123].sent[0][1], [[6]])
    np.testing.assert_array_equal(
        connections[123].sent[1][1], [[8], [10], [12]]
    )
    assert result["combined_batch_counts"] == {3: 2}
    assert result["group_sizes"] == {1: 1, 2: 1}
    assert result["chain_seconds"] == {42: 1.0, 123: 2.0}
    if timing:
        detail = result["coordination"]
        assert len(detail["timeline"]) == 2
        assert detail["timeline"][1]["requests"][0]["seed"] == 123
        parts = sum(
            result[k]
            for k in [
                "evaluation_seconds",
                "receive_wait_and_decode_seconds",
                "send_encode_seconds",
            ]
        )
        assert parts + detail["other_seconds"] == pytest.approx(
            detail["serve_wall_seconds"]
        )
        for row in detail["timeline"]:
            assert (
                row["model_start"]
                <= row["model_end"]
                <= row["send_start"]
                <= row["send_end"]
            )
    else:
        assert "coordination" not in result

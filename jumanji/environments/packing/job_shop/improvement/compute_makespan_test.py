# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.packing.job_shop.improvement.compute_makespan import (
    compute_earliest_start_times_and_makespan,
    compute_est_lst_makespan,
    compute_latest_start_times,
)

# Hardcoded max number of edges for the example matrices
MAX_NUM_EDGES = 50


class TestComputeMakespanJraph:
    """Test suite for the compute_earliest_start_times_and_makespan,
    compute_latest_start_times and compute_est_lst_makespan functions."""

    @pytest.fixture
    def matrices_example(self) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Example matrices for the forward and backward pass.
        There are 8 valid operations, from 0 to 7. Machines are given by ops_machines_ids.
        adj_mat_mc corresponds to the following operation order on machines:
        - Machine 0: 0 -> 3
        - Machine 1: 1 -> 6 -> 5
        - Machine 2: 4 -> 2 -> 7
        """

        # For indication
        ops_machine_ids = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, -1],
            ],
            jnp.int32,
        )
        del ops_machine_ids

        ops_durations = jnp.array(
            [
                [3, 2, 2],
                [2, 1, 4],
                [4, 3, -1],
            ],
            jnp.int32,
        )

        adj_mat_pc = jnp.array(
            [
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        # Machine constraint matrix
        adj_mat_mc = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        return ops_durations, adj_mat_pc, adj_mat_mc

    def test_compute_earliest_start_times_and_makespan__env_test_example(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test of forward pass using the example matrices."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example

        # Combine matrices to obtain the adjacency matrix of the disjunctive graph
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)

        est, makespan = compute_earliest_start_times_and_makespan(
            adj_mat, ops_durations, MAX_NUM_EDGES
        )

        # Verify source node has earliest start time = 0
        assert est[0] == 0

        # Verify target node has correct makespan, equal to 13.
        assert makespan == 13

    def test_compute_earliest_start_times_and_makespan__simple_chain(self) -> None:
        """Test forward pass on a simple chain graph, with one job and 3 operations."""
        # Create a simple chain graph: 0 -> 1 -> 2 -> 3
        adj_mat = jnp.array(
            [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
        )  # one job, one machine

        ops_durations = jnp.array([2.0, 3.0, 1.0])

        est, makespan = compute_earliest_start_times_and_makespan(
            adj_mat, ops_durations, MAX_NUM_EDGES
        )

        # Expected earliest start times:
        # Node 0 (source): 0
        # Node 1: 0
        # Node 2: 2
        # Node 3: 5
        # Node 4 (target): 6

        expected_est = jnp.array([0.0, 0.0, 2.0, 5.0, 6.0])
        assert jnp.allclose(est, expected_est)
        assert makespan == 6

    def test_compute_earliest_start_times_and_makespan__jit(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Confirm that the forward pass is only compiled once when jitted."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example
        # Combine matrices and set processing times
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)

        # JIT the function
        jitted_fn = jax.jit(
            chex.assert_max_traces(compute_earliest_start_times_and_makespan, n=1),
            static_argnums=(2,),
        )

        # First call (compilation)
        est1, makespan1 = jitted_fn(adj_mat, ops_durations, MAX_NUM_EDGES)

        # Second call (should use cached compilation)
        est2, makespan2 = jitted_fn(adj_mat, ops_durations, MAX_NUM_EDGES)

        assert jnp.allclose(est1, est2)
        assert makespan1 == makespan2
        assert makespan1 == 13

    def test_compute_latest_start_times__env_test_example(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test forward pass using the same example as in conftest DummyGenerator."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example

        # Combine matrices and set processing times
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)

        makespan = 13
        lst = compute_latest_start_times(adj_mat, ops_durations, makespan, MAX_NUM_EDGES)

        # Verify source node has lst = 0 and last node has lst = makespan
        assert lst[0] == 0
        assert lst[-1] == 13

        # Verify each operation has correct latest start time
        assert jnp.all(lst[1:9] == jnp.array([0, 3, 8, 5, 7, 9, 5, 10]))

    def test_compute_latest_start_times__simple_chain(self) -> None:
        """Test backward pass on a simple chain graph, with one job and 3 operations."""
        # Create a simple chain graph: 0 -> 1 -> 2 -> 3
        adj_mat = jnp.array(
            [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
        )  # one job, one machine

        ops_durations = jnp.array([2.0, 3.0, 1.0])

        makespan = 6
        lst = compute_latest_start_times(adj_mat, ops_durations, makespan, MAX_NUM_EDGES)

        # Expected latest start times:
        # Node 0 (source): 0
        # Node 1: 0
        # Node 2: 2
        # Node 3: 5
        # Node 4 (target): 6

        expected_lst = jnp.array([0.0, 0.0, 2.0, 5.0, 6.0])
        assert jnp.allclose(lst, expected_lst)

    def test_compute_latest_start_times__jit(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test that backward pass can be jitted and only compiled once."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example

        # Combine matrices
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)

        makespan = 13

        jitted_fn = jax.jit(
            chex.assert_max_traces(compute_latest_start_times, n=1), static_argnums=(3,)
        )

        # First call (compilation)
        lst1 = jitted_fn(adj_mat, ops_durations, makespan, MAX_NUM_EDGES)

        # Second call (should use cached compilation)
        lst2 = jitted_fn(adj_mat, ops_durations, makespan, MAX_NUM_EDGES)

        assert jnp.allclose(lst1, lst2)

        # Verify sink node has lst = makespan
        assert lst1[-1] == makespan

        # Verify source node has lst >= 0
        assert lst1[0] >= 0

    def test_compute_est_lst_makespan__env_test_example(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test forward and backward pass using the example matrices."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example

        # Combine matrices
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)

        est, lst, makespan = compute_est_lst_makespan(adj_mat, ops_durations, MAX_NUM_EDGES)

        # Verify source node has est = 0
        assert est[0] == 0

        # Verify target node has makespan = 13
        assert makespan == 13

    def test_compute_est_lst_makespan__simple_chain(self) -> None:
        """Test forward and backward pass on a simple chain graph, with one job and 3 operations."""
        # Create a simple chain graph: 0 -> 1 -> 2 -> 3
        adj_mat = jnp.array(
            [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
        )  # one job, one machine

        ops_durations = jnp.array([2.0, 3.0, 1.0])

        est, lst, makespan = compute_est_lst_makespan(adj_mat, ops_durations, MAX_NUM_EDGES)

        # Expected earliest start times:
        # Node 0 (source): 0
        # Node 1: 0
        # Node 2: 2
        # Node 3: 5
        # Node 4 (target): 6

        expected_est = jnp.array([0.0, 0.0, 2.0, 5.0, 6.0])
        expected_lst = jnp.array([0.0, 0.0, 2.0, 5.0, 6.0])

        assert jnp.allclose(est, expected_est)
        assert jnp.allclose(lst, expected_lst)

    def test_compute_est_lst_makespan__jit(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test that forward and backward pass can be jitted and only compiled once."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example

        # Combine matrices
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)

        jitted_fn = jax.jit(
            chex.assert_max_traces(compute_est_lst_makespan, n=1), static_argnums=(2,)
        )

        # First call (compilation)
        est1, lst1, makespan1 = jitted_fn(adj_mat, ops_durations, MAX_NUM_EDGES)

        # Second call (should use cached compilation)
        est2, lst2, makespan2 = jitted_fn(adj_mat, ops_durations, MAX_NUM_EDGES)

        assert jnp.allclose(est1, est2)
        assert jnp.allclose(lst1, lst2)
        assert makespan1 == makespan2

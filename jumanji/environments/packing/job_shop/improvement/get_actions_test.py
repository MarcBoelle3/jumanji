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
    compute_est_lst_makespan,
)
from jumanji.environments.packing.job_shop.improvement.get_actions import (
    get_action_mask_n5,
    get_critical_operations,
)

# Hardcoded max number of edges for the example matrices
MAX_NUM_EDGES = 50


class TestGetCriticalOperations:
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

    def test_get_critical_operations__dummy_instance(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test that get_critical_operations correctly identifies critical operations
        in the dummy instance."""

        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example
        max_num_jobs, max_num_ops = ops_durations.shape

        # Compute est and lst
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, _ = compute_est_lst_makespan(adj_mat, ops_durations, MAX_NUM_EDGES)

        # Get critical operations
        critical_ops = get_critical_operations(
            est, lst, adj_mat_mc, ops_durations, max_num_jobs, max_num_ops, MAX_NUM_EDGES
        )

        # Check that source and target nodes are not included
        assert critical_ops.shape == (max_num_jobs * max_num_ops, 8)

        is_on_critical_path = critical_ops[:, 0]
        num_critical_block = critical_ops[:, 1]
        is_left_critical_block = critical_ops[:, 2]
        is_right_critical_block = critical_ops[:, 3]
        left_neighbor_idx = critical_ops[:, 4]
        right_neighbor_idx = critical_ops[:, 5]
        left_end_critical_block = critical_ops[:, 6]
        right_end_critical_block = critical_ops[:, 7]
        jax.debug.print("critical block info: {}", critical_ops)

        # Critical operations should be operations 0, 1, 5, 6
        assert jnp.all(is_on_critical_path == jnp.array([1, 1, 0, 0, 0, 1, 1, 0, 0]))

        # Operations 1, 5, 6 are on the same critical block, the others are on their own
        assert jnp.all(num_critical_block == jnp.array([0, 1, 2, 3, 4, 1, 1, 7, 8]))

        # Indicates if the operation are the left end of the critical block
        assert jnp.all(is_left_critical_block == jnp.array([1, 1, 1, 1, 1, 0, 0, 1, 1]))

        # Indicates if the operation are the right end of the critical block
        assert jnp.all(is_right_critical_block == jnp.array([1, 0, 1, 1, 1, 1, 0, 1, 1]))

        # Indicates the index of the left neighbor of the operation, padded with -1 for
        # non-critical operations
        assert jnp.all(left_neighbor_idx == jnp.array([-1, -1, -1, -1, -1, 6, 1, -1, -1]))

        # Indicates the index of the right neighbor of the operation, padded with -1 for
        # non-critical operations
        assert jnp.all(right_neighbor_idx == jnp.array([-1, 6, -1, -1, -1, -1, 5, -1, -1]))

        # Indicates the index of the left end of the critical block
        assert jnp.all(left_end_critical_block == jnp.array([0, 1, 2, 3, 4, 1, 1, 7, 8]))

        # Indicates the index of the right end of the critical block
        assert jnp.all(right_end_critical_block == jnp.array([-1, 5, -1, -1, -1, 5, 5, -1, -1]))

    def test_get_critical_operations_jit(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test that get_critical_operations is jit-able."""
        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example
        max_num_jobs, max_num_ops = ops_durations.shape

        # Compute est and lst
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, _ = compute_est_lst_makespan(adj_mat, ops_durations, MAX_NUM_EDGES)

        call_fn = jax.jit(
            chex.assert_max_traces(get_critical_operations, n=1), static_argnums=(4, 5, 6)
        )

        critical_ops = call_fn(
            est, lst, adj_mat_mc, ops_durations, max_num_jobs, max_num_ops, MAX_NUM_EDGES
        )

        critical_ops2 = get_critical_operations(
            est, lst, adj_mat_mc, ops_durations, max_num_jobs, max_num_ops, MAX_NUM_EDGES
        )

        assert jnp.all(critical_ops == critical_ops2)

    def test_get_action_mask_n5(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test that get_action_mask_N5 correctly identifies valid actions."""
        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example
        max_num_jobs, max_num_ops = ops_durations.shape

        # Get critical operations
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, _ = compute_est_lst_makespan(adj_mat, ops_durations, MAX_NUM_EDGES)
        critical_ops = get_critical_operations(
            est, lst, adj_mat_mc, ops_durations, max_num_jobs, max_num_ops, MAX_NUM_EDGES
        )

        # Get action mask
        action_mask = get_action_mask_n5(critical_ops, max_num_ops)

        # Verify action mask
        assert action_mask.shape == (max_num_jobs * max_num_ops, 2)
        jax.debug.print("Action mask N5: {}", action_mask)

        assert jnp.all(
            action_mask
            == jnp.array(
                [
                    [False, False],
                    [False, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            )
        )

    def test_get_action_mask_n5_jit(
        self, matrices_example: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> None:
        """Test that get_action_mask_N5 is jit-able."""
        ops_durations, adj_mat_pc, adj_mat_mc = matrices_example
        max_num_jobs, max_num_ops = ops_durations.shape

        # Get critical operations
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, _ = compute_est_lst_makespan(adj_mat, ops_durations, MAX_NUM_EDGES)
        critical_ops = get_critical_operations(
            est, lst, adj_mat_mc, ops_durations, max_num_jobs, max_num_ops, MAX_NUM_EDGES
        )

        call_fn = jax.jit(chex.assert_max_traces(get_action_mask_n5, n=1), static_argnums=(1,))

        action_mask = call_fn(critical_ops, max_num_ops)
        action_mask2 = get_action_mask_n5(critical_ops, max_num_ops)

        assert jnp.all(action_mask == action_mask2)

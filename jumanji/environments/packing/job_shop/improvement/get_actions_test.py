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

"""Comprehensive tests for get_actions.py functions."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.packing.job_shop.improvement.compute_makespan import (
    compute_est_lst_makespan,
)
from jumanji.environments.packing.job_shop.improvement.get_actions import (
    CBFields,
    get_action_mask_n5,
    get_action_mask_n6,
    get_critical_operations_features,
    select_operations_to_switch,
)


class TestFixtures:
    """Test fixtures providing common test data."""

    # Constants for test matrices
    MAX_NUM_EDGES: int = 50
    MAX_NUM_JOBS: int = 3
    MAX_NUM_OPS: int = 3

    @pytest.fixture
    def simple_job_shop_instance(self) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Simple job shop instance with 3 jobs and 3 operations each.

        Job structure:
        - Job 0: ops [0,1,2] on machines [0,1,2] with durations [3,2,2]
        - Job 1: ops [3,4,5] on machines [0,2,1] with durations [2,1,4]
        - Job 2: ops [6,7] on machines [1,2] with durations [4,3]

        Machine scheduling:
        - Machine 0: 0 -> 3
        - Machine 1: 1 -> 6 -> 5
        - Machine 2: 4 -> 2 -> 7

        Returns:
            Tuple of (ops_durations, adj_mat_pc, adj_mat_mc, num_ops_per_job)
        """

        # Only for indication
        ops_machine_ids = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, -1],
            ],
            dtype=jnp.int32,
        )
        del ops_machine_ids

        ops_durations = jnp.array(
            [
                [3, 2, 2],
                [2, 1, 4],
                [4, 3, -1],
            ],
            dtype=jnp.float32,
        )

        # Precedence constraints (job ordering)
        adj_mat_pc = jnp.array(
            [
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # source -> ops 0,3,6
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 -> 1
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],  # 1 -> 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],  # 2 -> target
                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],  # 3 -> 4
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4 -> 5
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 5 -> target
                [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],  # 6 -> 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],  # 7 -> target
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # target
            ],
            dtype=jnp.float32,
        )

        # Machine constraints
        adj_mat_mc = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # source
                [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],  # 0 -> 3 (machine 0)
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # 1 -> 6 (machine 1)
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],  # 2 -> 7 (machine 2)
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 4 -> 2 (machine 2)
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],  # 6 -> 5 (machine 1)
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # target
            ],
            dtype=jnp.int32,
        )

        num_ops_per_job = jnp.array([3, 3, 2], dtype=jnp.int32)

        # Only for indication
        scheduled_times = jnp.array([[0.0, 3.0, 6.0], [3.0, 5.0, 9.0], [5.0, 9.0, -jnp.inf]])
        del scheduled_times

        return ops_durations, adj_mat_pc, adj_mat_mc, num_ops_per_job

    @pytest.fixture
    def computed_schedule_data(
        self, simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Compute EST, LST, and makespan for the simple job shop instance.

        Returns:
            Tuple of (est, lst, makespan)
        """
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, makespan = compute_est_lst_makespan(adj_mat, ops_durations, self.MAX_NUM_EDGES)
        return est, lst, makespan


class TestGetCriticalOperationsFeatures(TestFixtures):
    """Test suite for get_critical_operations_features function."""

    def test_basic_functionality(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test basic functionality of get_critical_operations_features."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est=est,
            lst=lst,
            adj_mat_mc=adj_mat_mc,
            ops_durations=ops_durations,
            max_num_jobs=self.MAX_NUM_JOBS,
            max_num_ops=self.MAX_NUM_OPS,
            max_num_edges=self.MAX_NUM_EDGES,
        )

        # Verify output shape and type
        expected_cb_shape = (self.MAX_NUM_JOBS * self.MAX_NUM_OPS, 8)
        expected_gap_shape = (self.MAX_NUM_JOBS * self.MAX_NUM_OPS, 2)
        assert critical_block_info.shape == expected_cb_shape
        assert gap_left_right.shape == expected_gap_shape
        assert critical_block_info.dtype == jnp.int32
        assert gap_left_right.dtype == jnp.float32

        # Verify critical path identification
        is_on_critical_path = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH]
        # Operations 0, 1, 5, 6 should be on critical path based on the schedule
        expected_critical = jnp.array([1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=jnp.int32)
        assert jnp.array_equal(is_on_critical_path, expected_critical)

    def test_critical_block_structure(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test that critical blocks are correctly identified and structured."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        block_ids = critical_block_info[:, CBFields.BLOCK_ID]
        is_left = critical_block_info[:, CBFields.IS_LEFT]
        is_right = critical_block_info[:, CBFields.IS_RIGHT]

        # 1->6->5 is represented as critical block n°1
        expected_block_ids = jnp.array([0, 1, 2, 3, 4, 1, 1, 7, 8], dtype=jnp.int32)
        assert jnp.array_equal(block_ids, expected_block_ids)

        # Verify left/right end identification
        expected_is_left = jnp.array([1, 1, 1, 1, 1, 0, 0, 1, 1], dtype=jnp.int32)
        expected_is_right = jnp.array([1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=jnp.int32)
        assert jnp.array_equal(is_left, expected_is_left)
        assert jnp.array_equal(is_right, expected_is_right)

    def test_gap_computation(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test that time gaps are correctly computed."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        gap_left = gap_left_right[:, 0]
        gap_right = gap_left_right[:, 1]

        # Gap is -1 for dummy operations
        expected_gap_left = jnp.array([0, 3, 0, 0, 5, 0, 0, 1, -1], dtype=jnp.float32)
        expected_gap_right = jnp.array([0, 0, 1, 8, 0, 0, 0, 1, -1], dtype=jnp.float32)
        assert jnp.array_equal(gap_left, expected_gap_left)
        assert jnp.array_equal(gap_right, expected_gap_right)

        # Gaps should be non-negative for valid operations
        valid_ops_mask = ops_durations.flatten() >= 0
        assert jnp.all(gap_left[valid_ops_mask] >= -1)  # -1 for non-critical ops
        assert jnp.all(gap_right[valid_ops_mask] >= -1)

    def test_jit_compilation(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test that the function can be JIT compiled and only compiled once for repeated calls."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        chex.clear_trace_counter()
        jit_fn = jax.jit(
            chex.assert_max_traces(get_critical_operations_features, n=1), static_argnums=(4, 5, 6)
        )

        result_jit_cb_1, result_jit_gap_1 = jit_fn(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )
        result_jit_cb_2, result_jit_gap_2 = jit_fn(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        result_normal_cb, result_normal_gap = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        assert jnp.array_equal(result_jit_cb_1, result_normal_cb)
        assert jnp.allclose(result_jit_gap_1, result_normal_gap)
        assert jnp.array_equal(result_jit_cb_2, result_normal_cb)
        assert jnp.allclose(result_jit_gap_2, result_normal_gap)

    def test_edge_case(self) -> None:
        """Test edge case with only one valid operation."""
        # Test with minimal valid input
        ops_durations = jnp.array([[1.0, -1, -1]], dtype=jnp.float32)

        # Simple adjacency matrices for single operation
        adj_mat_mc = jnp.zeros((5, 5), dtype=jnp.int32)  # source, op, padding, target
        est = jnp.array(
            [0.0, 0.0, -jnp.inf, -jnp.inf, 1.0], dtype=jnp.float32
        )  # source, op, padding, target
        lst = jnp.array([0.0, 0.0, jnp.inf, jnp.inf, 1.0], dtype=jnp.float32)

        result_cb, result_gap = get_critical_operations_features(
            est, lst, adj_mat_mc, ops_durations, 1, 3, 10
        )

        assert result_cb.shape == (3, 8)
        assert result_gap.shape == (3, 2)
        # First operation should be critical
        assert result_cb[0, CBFields.IS_ON_CRITICAL_PATH] == 1


class TestGetActionMaskN5(TestFixtures):
    """Test suite for get_action_mask_n5 function."""

    def test_basic_functionality(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test basic functionality of get_action_mask_n5."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        action_mask = get_action_mask_n5(critical_block_info, self.MAX_NUM_OPS)

        # Verify output shape and type
        expected_shape = (self.MAX_NUM_JOBS * self.MAX_NUM_OPS, 2)
        assert action_mask.shape == expected_shape
        assert action_mask.dtype == bool

        # Verify that only operations on the edge of critical blocks have valid actions
        is_left = critical_block_info[:, CBFields.IS_LEFT].astype(bool)
        is_right = critical_block_info[:, CBFields.IS_RIGHT].astype(bool)
        is_left_or_right = jnp.broadcast_to(is_left[:, None] | is_right[:, None], action_mask.shape)
        assert jnp.all(is_left_or_right[action_mask])

        ## Verify, in this special case, that the only valid actions are
        # 1 : move right, 5: move left
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

    def test_jit_compilation(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Confirm that get_action_mask_n5 is only compiled once when jitted."""
        chex.clear_trace_counter()
        jit_fn = jax.jit(chex.assert_max_traces(get_action_mask_n5, n=1), static_argnums=(1,))
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        # Call twice to check it does not compile twice
        result1 = jit_fn(critical_block_info, self.MAX_NUM_OPS)
        result2 = jit_fn(critical_block_info, self.MAX_NUM_OPS)
        assert jnp.array_equal(result1, result2)

    def test_empty_critical_blocks(self) -> None:
        """Test behavior when there are no critical blocks."""
        # Create critical block info with no critical operations
        critical_block_info = jnp.zeros((9, 8), dtype=jnp.int32)

        action_mask = get_action_mask_n5(critical_block_info, 3)

        # Should have no valid actions
        assert not jnp.any(action_mask)


class TestGetActionMaskN6(TestFixtures):
    """Test suite for get_action_mask_n6 function."""

    def test_basic_functionality(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test basic functionality of get_action_mask_n6."""
        ops_durations, adj_mat_pc, adj_mat_mc, num_ops_per_job = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        action_mask = get_action_mask_n6(
            critical_block_info, self.MAX_NUM_OPS, est, num_ops_per_job
        )

        # Verify output shape and type
        expected_shape = (self.MAX_NUM_JOBS * self.MAX_NUM_OPS, 2)
        assert action_mask.shape == expected_shape
        assert action_mask.dtype == bool

        # In this case, the only valid actions are 1: move right, 6 : move left/right, 5: move right
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
                    [True, True],
                    [False, False],
                    [False, False],
                ]
            )
        )

    def test_jit_compilation(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test that get_action_mask_n6 can be JIT compiled and only traced once."""
        ops_durations, adj_mat_pc, adj_mat_mc, num_ops_per_job = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        chex.clear_trace_counter()
        traced_fn = chex.assert_max_traces(get_action_mask_n6, n=1)
        jit_fn = jax.jit(traced_fn, static_argnums=(1,))

        result_jit = jit_fn(critical_block_info, self.MAX_NUM_OPS, est, num_ops_per_job)
        result_normal = get_action_mask_n6(
            critical_block_info, self.MAX_NUM_OPS, est, num_ops_per_job
        )

        assert jnp.array_equal(result_jit, result_normal)

        # Call again to check it does not trace twice
        result_jit2 = jit_fn(critical_block_info, self.MAX_NUM_OPS, est, num_ops_per_job)
        assert jnp.array_equal(result_jit, result_jit2)

    def test_two_operation_block_constraint(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test special handling of critical blocks with only 2 operations."""
        ops_durations, adj_mat_pc, adj_mat_mc, num_ops_per_job = simple_job_shop_instance
        # Remove operation 5 from the schedule:
        ops_durations = ops_durations.at[1, 2].set(
            -1
        )  # put operation 2 of job 1 to -1 (dummy operation)
        est, lst, _ = computed_schedule_data
        est = est.at[5 + 1].set(-jnp.inf)  # +1 because of source node
        lst = lst.at[5 + 1].set(jnp.inf)  # +1 because of source node

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        action_mask = get_action_mask_n6(
            critical_block_info, self.MAX_NUM_OPS, est, num_ops_per_job
        )

        # For 2-operation blocks, only left operation should be allowed to move
        # Here, 1 can move right and 6 can move left, but we only authorize 1 to move right
        assert jnp.all(
            action_mask
            == jnp.array(
                [
                    [False, False],
                    [False, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            )
        )


class TestSelectOperationsToSwitch(TestFixtures):
    """Test suite for select_operations_to_switch function."""

    def test_n5_operation_selection(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test operation selection for N5 neighborhood."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        # Test left move (direction 0)
        action = jnp.array([5, 0], dtype=jnp.int32)  # op 5, direction left
        result = select_operations_to_switch(critical_block_info, action, neighborhood=5)

        assert result.shape == (3,)
        # Should return [neighbor_idx, chosen_op_idx, direction]
        # For left move, neighbor comes before chosen operation
        assert result[1] == 5  # chosen operation
        assert result[2] == 0  # direction

        # Test right move (direction 1)
        action = jnp.array([1, 1], dtype=jnp.int32)  # op 1, direction right
        result = select_operations_to_switch(critical_block_info, action, neighborhood=5)

        assert result.shape == (3,)
        assert result[0] == 1  # chosen operation comes first
        assert result[2] == 1  # direction

    def test_n6_operation_selection(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test operation selection for N6 neighborhood."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        # Test moving to left end (direction 0)
        action = jnp.array([6, 0], dtype=jnp.int32)  # op 6, move to left end
        result = select_operations_to_switch(critical_block_info, action, neighborhood=6)

        assert result.shape == (3,)
        # Should return operation to move to left end of its critical block
        left_end_idx = critical_block_info[6, CBFields.LEFT_END]
        assert result[1] == 6  # chosen operation
        assert result[0] == left_end_idx  # left end of block

        # Test moving to right end (direction 1)
        action = jnp.array([1, 1], dtype=jnp.int32)  # op 1, move to right end
        result = select_operations_to_switch(critical_block_info, action, neighborhood=6)

        assert result.shape == (3,)
        right_end_idx = critical_block_info[1, CBFields.RIGHT_END]
        assert result[0] == 1  # chosen operation
        assert result[1] == right_end_idx  # right end of block

    def test_jit_compilation(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test that select_operations_to_switch can be JIT compiled
        and only compiled once for repeated calls."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, _ = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        action = jnp.array([1, 1], dtype=jnp.int32)

        chex.clear_trace_counter()
        jit_fn = jax.jit(
            chex.assert_max_traces(select_operations_to_switch, n=1), static_argnums=(2,)
        )

        result_jit_1 = jit_fn(critical_block_info, action, 5)
        result_jit_2 = jit_fn(critical_block_info, action, 5)
        result_normal = select_operations_to_switch(critical_block_info, action, neighborhood=5)

        assert jnp.array_equal(result_jit_1, result_normal)
        assert jnp.array_equal(result_jit_2, result_normal)

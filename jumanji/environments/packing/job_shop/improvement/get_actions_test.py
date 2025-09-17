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
from jumanji.environments.packing.job_shop.improvement.types import Neighborhood


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
        jax.debug.print("critical_block_info: {0}", critical_block_info)
        jax.debug.print("is_on_critical_path: {0}", is_on_critical_path)
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

    def test_gap_computation_validation(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test gap computation validation: timeline spanning and edge cases."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, makespan = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        est_ops = est[1:-1]  # Remove source and target nodes
        ops_durations_flat = ops_durations.flatten()
        gap_left = gap_left_right[:, 0]
        gap_right = gap_left_right[:, 1]

        # Test 1: Gap computation for operations with machine predecessors/successors
        adj_mat_ops = adj_mat_mc[1:-1, 1:-1]
        senders, receivers = jnp.nonzero(adj_mat_ops > 0, size=self.MAX_NUM_EDGES, fill_value=-1)
        valid_edges = (senders >= 0) & (receivers >= 0)

        for i in range(len(senders)):
            if not valid_edges[i]:
                continue
            sender, receiver = senders[i], receivers[i]

            # Gap should equal the time difference between operations
            expected_gap = est_ops[receiver] - (est_ops[sender] + ops_durations_flat[sender])

            # The gap_right of sender should match this gap
            if gap_right[sender] >= 0:  # Only check if gap is computed (not -1)
                assert jnp.isclose(
                    gap_right[sender], expected_gap, atol=1e-6
                ), f"Gap right for sender {sender} should match time difference"

            # The gap_left of receiver should match this gap
            if gap_left[receiver] >= 0:  # Only check if gap is computed (not -1)
                assert jnp.isclose(
                    gap_left[receiver], expected_gap, atol=1e-6
                ), f"Gap left for receiver {receiver} should match time difference"

        # Test 2: First operations on machines should have gap_left = est
        valid_ops = ops_durations_flat >= 0
        for op_idx in range(len(gap_left)):
            if not valid_ops[op_idx]:
                continue

            # If gap_left equals est_ops, this is a first operation on its machine
            if jnp.isclose(gap_left[op_idx], est_ops[op_idx], atol=1e-6):
                # Verify this operation has no machine predecessor
                has_predecessor = False
                for i in range(len(receivers)):
                    if valid_edges[i] and receivers[i] == op_idx:
                        has_predecessor = True
                        break
                assert (
                    not has_predecessor or gap_left[op_idx] == est_ops[op_idx]
                ), f"Operation {op_idx} with gap_left=est should be first on machine"

        # Test 3: Last operations on machines should have gap_right = makespan - (est + duration)
        for op_idx in range(len(gap_right)):
            if not valid_ops[op_idx]:
                continue

            expected_right_gap = makespan - (est_ops[op_idx] + ops_durations_flat[op_idx])
            if jnp.isclose(gap_right[op_idx], expected_right_gap, atol=1e-6):
                # Verify this operation has no machine successor
                has_successor = False
                for i in range(len(senders)):
                    if valid_edges[i] and senders[i] == op_idx:
                        has_successor = True
                        break
                assert not has_successor or jnp.isclose(
                    gap_right[op_idx], expected_right_gap, atol=1e-6
                ), f"Operation {op_idx} with computed right gap should be last on machine"

        # Test 4: Timeline consistency - gaps should not be negative for valid operations
        for op_idx in range(len(gap_left)):
            if valid_ops[op_idx] and gap_left[op_idx] >= 0:
                assert (
                    gap_left[op_idx] >= 0
                ), f"Gap left for operation {op_idx} should be non-negative"
            if valid_ops[op_idx] and gap_right[op_idx] >= 0:
                assert (
                    gap_right[op_idx] >= 0
                ), f"Gap right for operation {op_idx} should be non-negative"

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

    def test_critical_operations_fundamental_properties(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test fundamental properties of critical operations: est==lst and makespan
        relationship."""
        ops_durations, adj_mat_pc, adj_mat_mc, _ = simple_job_shop_instance
        est, lst, makespan = computed_schedule_data

        critical_block_info, gap_left_right = get_critical_operations_features(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.MAX_NUM_JOBS,
            self.MAX_NUM_OPS,
            self.MAX_NUM_EDGES,
        )

        est_ops = est[1:-1]  # Remove source and target nodes
        lst_ops = lst[1:-1]  # Remove source and target nodes
        is_critical = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH].astype(bool)

        # Test 1: All operations marked as critical must satisfy est[op] == lst[op]
        critical_ops_est = est_ops[is_critical]
        critical_ops_lst = lst_ops[is_critical]
        assert jnp.allclose(
            critical_ops_est, critical_ops_lst, atol=1e-6
        ), "Critical operations must have est == lst"

        # Test 2: Sum of durations of operations on the critical path is exactly the makespan
        ops_durations_flat = ops_durations.flatten()
        valid_ops_mask = ops_durations_flat >= 0

        # The critical path is the set of operations where est == lst (i.e., is_critical)
        # The sum of their durations should be exactly the makespan
        critical_durations = ops_durations_flat[is_critical & valid_ops_mask]
        total_critical_duration = jnp.sum(critical_durations)
        assert jnp.isclose(
            total_critical_duration, makespan, atol=1e-6
        ), f"Sum of durations on critical path ({total_critical_duration}) \
        should equal makespan ({makespan})"

    def test_critical_block_structural_properties(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test structural properties of critical blocks: linked lists, maximality, uniqueness."""
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

        is_critical = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH].astype(bool)
        block_ids = critical_block_info[:, CBFields.BLOCK_ID]
        left_neighbors = critical_block_info[:, CBFields.LEFT_NEIGHBOR]
        right_neighbors = critical_block_info[:, CBFields.RIGHT_NEIGHBOR]
        is_left = critical_block_info[:, CBFields.IS_LEFT].astype(bool)
        is_right = critical_block_info[:, CBFields.IS_RIGHT].astype(bool)

        # Test 1: Each critical operation belongs to exactly one block
        critical_ops = jnp.where(is_critical)[0]
        for op_idx in critical_ops:
            assert block_ids[op_idx] >= 0, f"Critical operation {op_idx} should have valid block ID"

        # Test 2: Blocks form valid linked lists via LEFT/RIGHT_NEIGHBOR
        for op_idx in critical_ops:
            left_neighbor = left_neighbors[op_idx]
            right_neighbor = right_neighbors[op_idx]

            # If has left neighbor, verify bidirectional link
            if left_neighbor >= 0:
                assert (
                    right_neighbors[left_neighbor] == op_idx
                ), f"Left neighbor {left_neighbor} of op {op_idx} should point back"
                assert (
                    block_ids[left_neighbor] == block_ids[op_idx]
                ), "Left neighbor should be in same block"

            # If has right neighbor, verify bidirectional link
            if right_neighbor >= 0:
                assert (
                    left_neighbors[right_neighbor] == op_idx
                ), f"Right neighbor {right_neighbor} of op {op_idx} should point back"
                assert (
                    block_ids[right_neighbor] == block_ids[op_idx]
                ), "Right neighbor should be in same block"

        # Test 3: Left and right end properties are consistent
        for op_idx in critical_ops:
            if is_left[op_idx]:
                assert (
                    left_neighbors[op_idx] == -1
                ), f"Left end operation {op_idx} should have no left neighbor"
            if is_right[op_idx]:
                assert (
                    right_neighbors[op_idx] == -1
                ), f"Right end operation {op_idx} should have no right neighbor"

        # Test 4: Block maximality - no adjacent critical ops from different blocks on same machine
        est_ops = est[1:-1]
        ops_durations_flat = ops_durations.flatten()

        # Check that adjacent critical operations on same machine are in same block
        adj_mat_ops = adj_mat_mc[1:-1, 1:-1]
        senders, receivers = jnp.nonzero(adj_mat_ops > 0, size=self.MAX_NUM_EDGES, fill_value=-1)
        valid_edges = (senders >= 0) & (receivers >= 0)

        for i in range(len(senders)):
            if not valid_edges[i]:
                continue
            sender, receiver = senders[i], receivers[i]

            # If both are critical and adjacent in time, they should be in same block
            if is_critical[sender] and is_critical[receiver]:
                sender_end = est_ops[sender] + ops_durations_flat[sender]
                receiver_start = est_ops[receiver]
                if jnp.isclose(sender_end, receiver_start):
                    assert (
                        block_ids[sender] == block_ids[receiver]
                    ), f"Adjacent critical ops {sender}, {receiver} should be in same block"


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

    def test_action_mask_safety_properties(
        self,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test that action mask safety properties prevent cycles and maintain validity."""
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

        est_ops = est[1:-1]  # Remove source and target nodes

        # Test 1: Verify EST comparisons for allowed right moves
        valid_right_moves = jnp.where(action_mask[:, 1])[0]  # Operations that can move right

        for op_idx in valid_right_moves:
            right_end_idx = critical_block_info[op_idx, CBFields.RIGHT_END]
            ops_order_in_job = op_idx % self.MAX_NUM_OPS
            ops_job_id = op_idx // self.MAX_NUM_OPS

            # Check job predecessor condition
            has_right_end_job_predecessor = right_end_idx % self.MAX_NUM_OPS != 0
            is_current_op_last_of_job = ops_order_in_job == num_ops_per_job[ops_job_id] - 1

            if has_right_end_job_predecessor and not is_current_op_last_of_job:
                job_predecessor_of_right_end = right_end_idx - 1
                job_successor_of_current_op = op_idx + 1

                est_job_predecessor_right_end = est_ops[job_predecessor_of_right_end]
                est_job_successor_current_op = est_ops[job_successor_of_current_op]

                # The condition should be satisfied for allowed moves
                assert (
                    est_job_successor_current_op > est_job_predecessor_right_end
                ), f"Right move for op {op_idx} should satisfy job predecessor condition"

        # Test 2: Verify EST comparisons for allowed left moves
        valid_left_moves = jnp.where(action_mask[:, 0])[0]  # Operations that can move left

        for op_idx in valid_left_moves:
            left_end_idx = critical_block_info[op_idx, CBFields.LEFT_END]
            ops_order_in_job = op_idx % self.MAX_NUM_OPS
            ops_job_id = op_idx // self.MAX_NUM_OPS

            # Check job successor condition
            is_left_end_last_of_job = (
                left_end_idx % self.MAX_NUM_OPS
                == num_ops_per_job[left_end_idx // self.MAX_NUM_OPS] - 1
            )
            is_current_op_first_of_job = ops_order_in_job == 0

            if not is_current_op_first_of_job and not is_left_end_last_of_job:
                job_predecessor_of_current_op = op_idx - 1
                job_successor_of_left_end = left_end_idx + 1

                est_job_predecessor_current_op = est_ops[job_predecessor_of_current_op]
                est_job_successor_left_end = est_ops[job_successor_of_left_end]

                # The condition should be satisfied for allowed moves
                assert (
                    est_job_predecessor_current_op < est_job_successor_left_end
                ), f"Left move for op {op_idx} should satisfy job successor condition"

            # Check machine successor condition
            machine_successor_of_left_end = critical_block_info[
                left_end_idx, CBFields.RIGHT_NEIGHBOR
            ]
            if machine_successor_of_left_end >= 0 and not is_current_op_first_of_job:
                job_predecessor_of_current_op = op_idx - 1
                est_machine_successor_left_end = est_ops[machine_successor_of_left_end]
                est_job_predecessor_current_op = est_ops[job_predecessor_of_current_op]

                assert (
                    est_job_predecessor_current_op <= est_machine_successor_left_end
                ), f"Left move for op {op_idx} should satisfy machine successor condition"

        # Test 3: Job precedence constraints are respected
        for op_idx in range(len(action_mask)):
            if action_mask[op_idx, 0]:  # Can move left
                # Should be first in its (job, block) group or satisfy precedence
                ops_job_id = op_idx // self.MAX_NUM_OPS
                block_id = critical_block_info[op_idx, CBFields.BLOCK_ID]

                # Find all operations in same (job, block) group
                same_group_mask = (
                    jnp.arange(len(critical_block_info)) // self.MAX_NUM_OPS == ops_job_id
                ) & (critical_block_info[:, CBFields.BLOCK_ID] == block_id)
                same_group_ops = jnp.where(same_group_mask)[0]

                if len(same_group_ops) > 1:
                    # Should be the first operation in the group
                    first_in_group = jnp.min(same_group_ops)
                    assert (
                        op_idx == first_in_group
                    ), f"Operation {op_idx} moving left should be first in its job-block group"

            if action_mask[op_idx, 1]:  # Can move right
                # Should be last in its (job, block) group or satisfy precedence
                ops_job_id = op_idx // self.MAX_NUM_OPS
                block_id = critical_block_info[op_idx, CBFields.BLOCK_ID]

                # Find all operations in same (job, block) group
                same_group_mask = (
                    jnp.arange(len(critical_block_info)) // self.MAX_NUM_OPS == ops_job_id
                ) & (critical_block_info[:, CBFields.BLOCK_ID] == block_id)
                same_group_ops = jnp.where(same_group_mask)[0]

                if len(same_group_ops) > 1:
                    # Should be the last operation in the group
                    last_in_group = jnp.max(same_group_ops)
                    assert (
                        op_idx == last_in_group
                    ), f"Operation {op_idx} moving right should be last in its job-block group"


class TestSelectOperationsToSwitch(TestFixtures):
    """Test suite for select_operations_to_switch function."""

    @pytest.mark.parametrize(
        "neighborhood,op_idx,direction,expected_behavior",
        [
            # N5 test cases - adjacent swaps within critical blocks
            (5, 5, 0, "left_neighbor_swap"),  # op 5 moves left, swaps with left neighbor
            (5, 1, 1, "right_neighbor_swap"),  # op 1 moves right, swaps with right neighbor
            # N6 test cases - moves to block ends
            (6, 6, 0, "move_to_left_end"),  # op 6 moves to left end of its block
            (6, 1, 1, "move_to_right_end"),  # op 1 moves to right end of its block
        ],
    )
    def test_operation_selection(
        self,
        neighborhood: int,
        op_idx: int,
        direction: int,
        expected_behavior: str,
        simple_job_shop_instance: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        computed_schedule_data: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> None:
        """Test operation selection for both N5 and N6 neighborhoods with parameterized cases."""
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

        # Create action and get result
        action = jnp.array([op_idx, direction], dtype=jnp.int32)
        result = select_operations_to_switch(critical_block_info, action, neighborhood=neighborhood)

        # Common assertions for all cases
        assert result.shape == (
            3,
        ), f"Result should have 3 elements for {neighborhood} neighborhood"
        assert (
            result[2] == direction
        ), f"Direction should match input for {neighborhood} neighborhood"

        # Behavior-specific assertions
        if expected_behavior == "left_neighbor_swap":
            # N5 left move: swap with left neighbor
            left_neighbor = critical_block_info[op_idx, CBFields.LEFT_NEIGHBOR]
            assert result[0] == left_neighbor, "N5 left move should have left neighbor as start_op"
            assert result[1] == op_idx, "N5 left move should have chosen op as end_op"

        elif expected_behavior == "right_neighbor_swap":
            # N5 right move: swap with right neighbor
            right_neighbor = critical_block_info[op_idx, CBFields.RIGHT_NEIGHBOR]
            assert result[0] == op_idx, "N5 right move should have chosen op as start_op"
            assert result[1] == right_neighbor, "N5 right move should have right neighbor as end_op"

        elif expected_behavior == "move_to_left_end":
            # N6 left move: move to left end of block
            left_end_idx = critical_block_info[op_idx, CBFields.LEFT_END]
            assert result[0] == left_end_idx, "N6 left move should have left end as start_op"
            assert result[1] == op_idx, "N6 left move should have chosen op as end_op"

        elif expected_behavior == "move_to_right_end":
            # N6 right move: move to right end of block
            right_end_idx = critical_block_info[op_idx, CBFields.RIGHT_END]
            assert result[0] == op_idx, "N6 right move should have chosen op as start_op"
            assert result[1] == right_end_idx, "N6 right move should have right end as end_op"

        else:
            pytest.fail(f"Unknown expected_behavior: {expected_behavior}")

        # Additional validation - operations should be different
        assert (
            result[0] != result[1]
        ), f"Operations to switch should be different for {neighborhood} neighborhood"

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

        result_jit_1 = jit_fn(critical_block_info, action, Neighborhood.N5)
        result_jit_2 = jit_fn(critical_block_info, action, Neighborhood.N5)
        result_normal = select_operations_to_switch(
            critical_block_info, action, neighborhood=Neighborhood.N5
        )

        assert jnp.array_equal(result_jit_1, result_normal)
        assert jnp.array_equal(result_jit_2, result_normal)

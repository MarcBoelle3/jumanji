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

from enum import IntEnum
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.improvement.types import Neighborhood


class CBFields(IntEnum):
    """Fields of the critical block information array."""

    IS_ON_CRITICAL_PATH = 0  # 1 if the operation is on the critical path, 0 otherwise
    BLOCK_ID = 1  # the number of the critical block the operation belongs to
    IS_LEFT = 2  # 1 if the operation is a left operation of a critical block, 0 otherwise
    IS_RIGHT = 3  # 1 if the operation is a right operation of a critical block, 0 otherwise
    LEFT_NEIGHBOR = 4  # the left neighbor of the operation
    RIGHT_NEIGHBOR = 5  # the right neighbor of the operation
    LEFT_END = 6  # operation index of the left end of the critical block
    RIGHT_END = 7  # operation index of the right end of the critical block


def _identify_critical_operations(
    est_ops: chex.Array,
    lst_ops: chex.Array,
) -> chex.Array:
    """Identify operations on the critical path.

    Critical operations have the property that est[op] == lst[op], meaning they
    cannot be moved in time without changing the makespan.

    Args:
        est_ops: Array of earliest start times (without source/target nodes)
        lst_ops: Array of latest start times (without source/target nodes)

    Returns:
        Boolean mask indicating which operations are critical
    """
    # Only consider operations with non-negative EST/LST values (valid operations)
    valid_ops = (est_ops >= 0) & (lst_ops >= 0)
    return valid_ops & jnp.isclose(est_ops, lst_ops)


def _find_and_sort_critical_edges(
    adj_mat_ops: chex.Array,
    critical_ops_mask: chex.Array,
    est_ops: chex.Array,
    ops_durations_flat: chex.Array,
    max_num_edges: jnp.int32,
) -> Tuple[chex.Array, chex.Array]:
    """Find machine constraint edges between critical operations and sort them
    by earliest start time of the first operation, so that the critical path
    will be scanned chronologically (needed for critical block features computation).

    Args:
        adj_mat_ops: Adjacency matrix (without source/target nodes)
        critical_ops_mask: Boolean mask of critical operations
        est_ops: Array of earliest start times
        ops_durations_flat: Flattened operation durations
        max_num_edges: Maximum number of edges

    Returns:
        Tuple of (senders, critical_block_pairs) where critical_block_pairs are sorted
    """
    # Find machine constraint edges
    senders, receivers = jnp.nonzero(adj_mat_ops > 0, size=max_num_edges, fill_value=-1)
    valid_edges_mask = (senders != -1) & (receivers != -1)

    # Check whether both operations are marked as critical
    is_critical_pair = critical_ops_mask[senders] & critical_ops_mask[receivers]

    # Check if the operations are adjacent in time (finish of one == start of the next)
    is_adjacent = jnp.isclose(est_ops[senders] + ops_durations_flat[senders], est_ops[receivers])

    # A pair is part of a critical block if:
    # (1) both ops are valid (not padded),
    # (2) both are on the critical path,
    # (3) they are adjacent in time.
    critical_blocks_mask = valid_edges_mask & is_critical_pair & is_adjacent

    # Sort pairs by earliest start time, with +inf for non-critical pairs
    filtered_est = jnp.where(
        critical_blocks_mask, est_ops[senders], jnp.inf
    )  # shape (max_num_edges,)
    senders_filtered = jnp.where(critical_blocks_mask, senders, -1)
    receivers_filtered = jnp.where(critical_blocks_mask, receivers, -1)
    sorted_edge_indices = jnp.argsort(filtered_est)
    critical_block_pairs = jnp.stack(
        [senders_filtered[sorted_edge_indices], receivers_filtered[sorted_edge_indices]], axis=-1
    )

    return senders, critical_block_pairs


def _initialize_critical_block_info(
    num_ops_total: jnp.int32,
    critical_ops_mask: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Initialize the critical block information structure.

    Args:
        num_ops_total: Total number of operations
        critical_ops_mask: Boolean mask of critical operations

    Returns:
        Tuple of (critical_block_info, right_end_array)
    """
    critical_block_info = jnp.zeros((num_ops_total, 8), dtype=jnp.int32)
    ops_indices = jnp.arange(num_ops_total)

    # Initialize all fields in one efficient operation
    critical_block_info = (
        critical_block_info.at[:, CBFields.IS_ON_CRITICAL_PATH]
        .set(critical_ops_mask)
        .at[:, CBFields.BLOCK_ID]
        .set(ops_indices)  # Each op starts as its own block
        .at[:, CBFields.LEFT_END]
        .set(ops_indices)  # Each op is initially its own left end
        .at[:, CBFields.IS_RIGHT]
        .set(1)  # All ops initially marked as right ends
        .at[:, CBFields.LEFT_NEIGHBOR]
        .set(-1)
        .at[:, CBFields.RIGHT_NEIGHBOR]
        .set(-1)
        .at[:, CBFields.RIGHT_END]
        .set(-1)
    )

    # Track right end of each critical block
    right_end_array = jnp.full(num_ops_total, -1, dtype=jnp.int32)

    return critical_block_info, right_end_array


def _find_critical_block_features(
    critical_block_info: chex.Array,
    right_end_array: chex.Array,
    critical_block_pairs: chex.Array,
    num_ops_total: jnp.int32,
) -> Tuple[chex.Array, chex.Array]:
    """Find critical block features by merging adjacent critical operations.

    Args:
        critical_block_info: Initial critical block information
        right_end_array: Array tracking right ends of blocks
        critical_block_pairs: Sorted pairs of adjacent critical operations
        num_ops_total: Total number of operations

    Returns:
        Tuple of (updated_critical_block_info, updated_right_end_array)
    """

    def stop_condition(loop_state: Tuple[jnp.int32, Tuple[chex.Array, chex.Array]]) -> jnp.bool_:
        """Stop when all valid critical block pairs have been processed."""
        i, _ = loop_state
        start_idx, end_idx = critical_block_pairs[i]
        return (i < num_ops_total) & (start_idx != -1) & (end_idx != -1)

    def merge_blocks_body(
        loop_state: Tuple[jnp.int32, Tuple[chex.Array, chex.Array]],
    ) -> Tuple[jnp.int32, Tuple[chex.Array, chex.Array]]:
        """Update the critical block information based on the critical block pair, by merging
           the critical block of the start_idx operation into the critical block
           of the end_idx operation.
           During one update, if the pair (start_idx, end_idx) is valid, we update the
           critical block information as follows:
           - num_critical_block of end_idx is set to the block_id of the start_idx operation.
           - left_end of end_idx is set to left_end of start_idx.
           - is_right of start_idx is set to 0.
           - right_neighbor of start_idx is set to the end_idx operation.
           - is_left of end_idx is set to 0.
           - left_neighbor of end_idx is set to the start_idx operation.

        Args:
            loop_state: Tuple containing (i, (critical_block_info, right_end_array))
                   with i being the current index in the critical_block_pairs array

        Returns:
            Tuple containing (updated critical_block_info, updated right_end_array)
        """

        i, (critical_block_info, right_end_array) = loop_state
        start_idx, end_idx = critical_block_pairs[i]

        # Get block information from start operation
        block_id = critical_block_info[start_idx, CBFields.BLOCK_ID].astype(jnp.int32)
        left_end_of_start_idx = critical_block_info[start_idx, CBFields.LEFT_END].astype(jnp.int32)

        critical_block_info = (
            critical_block_info.at[end_idx, CBFields.BLOCK_ID]
            .set(block_id)
            .at[end_idx, CBFields.LEFT_END]
            .set(left_end_of_start_idx)
            .at[end_idx, CBFields.LEFT_NEIGHBOR]
            .set(start_idx)
            .at[start_idx, CBFields.IS_RIGHT]
            .set(0)
            .at[start_idx, CBFields.RIGHT_NEIGHBOR]
            .set(end_idx)
        )

        # Set rightmost operation of the critical block to end_idx operation
        right_end_array = right_end_array.at[block_id].set(end_idx)

        return (i + 1, (critical_block_info, right_end_array))

    init_state = (0, (critical_block_info, right_end_array))
    _, (critical_block_info, right_end_array) = jax.lax.while_loop(
        stop_condition, merge_blocks_body, init_state
    )

    return critical_block_info, right_end_array


def _finalize_critical_blocks(
    critical_block_info: chex.Array,
    right_end_array: chex.Array,
    num_ops_total: jnp.int32,
) -> chex.Array:
    """Finalize critical block structure by setting right ends and identifying left operations.

    Args:
        critical_block_info: Critical block information after merging
        right_end_array: Array tracking right ends of blocks
        num_ops_total: Total number of operations

    Returns:
        Finalized critical block information
    """
    block_ids = critical_block_info[:, CBFields.BLOCK_ID]
    ops_indices = jnp.arange(num_ops_total)

    # Set right ends and identify left operations efficiently
    critical_block_info = (
        critical_block_info.at[:, CBFields.RIGHT_END]
        .set(jnp.where(block_ids != -1, right_end_array[block_ids], -1))
        .at[:, CBFields.IS_LEFT]
        .set(
            block_ids == ops_indices  # Left ops have block_id == their own index
        )
    )

    return critical_block_info


def _compute_time_gaps(
    est_ops: chex.Array,
    ops_durations_flat: chex.Array,
    senders: chex.Array,
    receivers: chex.Array,
    makespan: chex.Array,
    num_ops_total: jnp.int32,
) -> chex.Array:
    """Compute time gaps between operations on the same machine.

    Args:
        est_ops: Array of earliest start times
        ops_durations_flat: Flattened operation durations
        senders: Array of sender operation indices
        receivers: Array of receiver operation indices
        makespan: Total makespan of the schedule
        num_ops_total: Total number of operations

    Returns:
        Array of shape (num_ops_total, 2) with left and right gaps
    """
    # Find valid edges
    valid_edges_mask = (senders != -1) & (receivers != -1)

    # Compute gaps for all edges (including invalid ones, will be filtered later)
    sender_end_times = est_ops[senders] + ops_durations_flat[senders]
    receiver_start_times = est_ops[receivers]
    time_gaps = receiver_start_times - sender_end_times

    # Initialize gap arrays
    gap_left = jnp.full(num_ops_total, -1.0)
    gap_right = jnp.full(num_ops_total, -1.0)

    # Set gaps only for valid edges using scatter operations
    gap_left = gap_left.at[receivers].set(jnp.where(valid_edges_mask, time_gaps, -1))
    gap_right = gap_right.at[senders].set(jnp.where(valid_edges_mask, time_gaps, -1))

    # Handle first/last operations on each machine
    is_first_on_machine = (gap_left == -1.0) & (gap_right != -1.0)
    is_last_on_machine = (gap_left != -1.0) & (gap_right == -1.0)

    gap_left = jnp.where(is_first_on_machine, est_ops, gap_left)
    gap_right = jnp.where(is_last_on_machine, makespan - (est_ops + ops_durations_flat), gap_right)

    # Create separate gap array
    return jnp.stack([gap_left, gap_right], axis=-1)


def get_critical_operations_features(
    est: chex.Array,
    lst: chex.Array,
    adj_mat_mc: chex.Array,
    ops_durations: chex.Array,
    max_num_jobs: jnp.int32,
    max_num_ops: jnp.int32,
    max_num_edges: jnp.int32,
) -> chex.Array:
    """Get operations that are on the critical path, based on the earliest and latest start time
    of each operation. Critical operations have the property that est[op] == lst[op], ie they
    cannot be moved in time without changing the makespan.

    Documented in [doc.md](doc.md#core-algorithm-critical-block-identification).

    Args:
        est: array of earliest start times, shape (max_num_jobs * max_num_ops + 2,) including
             source and target nodes
        lst: array of latest start times, shape (max_num_jobs * max_num_ops + 2,) including
             source and target nodes
        adj_mat_mc: adjacency matrix of the machine constraint graph,
                    shape (max_num_jobs * max_num_ops + 2, max_num_jobs * max_num_ops + 2)
        ops_durations: array of operation durations, shape (max_num_jobs * max_num_ops,)

    Returns:
        Array of critical operations info of shape (num_ops_total, 8), see CBFields.
    """
    num_ops_total = max_num_ops * max_num_jobs

    # Input preprocessing
    makespan = est[-1]
    est_ops = est[1:-1]  # Remove source and target nodes
    lst_ops = lst[1:-1]  # Remove source and target nodes
    adj_mat_ops = adj_mat_mc[1:-1, 1:-1]  # Remove source and target nodes
    ops_durations_flat = ops_durations.reshape(-1)

    # 1. Critical operation identification
    critical_ops_mask = _identify_critical_operations(est_ops, lst_ops)

    # 2. Edge finding and sorting
    senders, critical_block_pairs = _find_and_sort_critical_edges(
        adj_mat_ops, critical_ops_mask, est_ops, ops_durations_flat, max_num_edges
    )

    # 3. Critical block initialization
    critical_block_info, right_end_array = _initialize_critical_block_info(
        num_ops_total, critical_ops_mask
    )

    # 4. Critical block features computation
    critical_block_info, right_end_array = _find_critical_block_features(
        critical_block_info, right_end_array, critical_block_pairs, num_ops_total
    )

    # 5. Block finalization
    critical_block_info = _finalize_critical_blocks(
        critical_block_info, right_end_array, num_ops_total
    )

    # 6. Time gap computation
    # We need to reconstruct receivers for gap computation
    receivers = jnp.nonzero(adj_mat_ops > 0, size=max_num_edges, fill_value=-1)[1]
    gap_left_right = _compute_time_gaps(
        est_ops, ops_durations_flat, senders, receivers, makespan, num_ops_total
    )

    return critical_block_info, gap_left_right


def get_action_mask_n5(critical_block_info: chex.Array, max_num_ops: int) -> chex.Array:
    """Get the mask of valid actions for the N5 neighborhood.

    Documented in [doc.md](doc.md#n5-neighborhood).

    Args:
        critical_block_info: array of critical block information, shape (num_ops_total, 8)
        max_num_ops: maximum number of operations per job

    Returns:
        A mask of valid actions, shape (num_ops_total, 2)
    """
    num_ops_total = critical_block_info.shape[0]

    # Obtain left and right ends of critical blocks
    is_critical = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH]
    is_left_end = critical_block_info[:, CBFields.IS_LEFT]
    is_right_end = critical_block_info[:, CBFields.IS_RIGHT]

    # Mask operations that are both left and right ends
    # (corresponds to critical block with one operation and no possible action)
    is_left_end_only = is_critical & is_left_end & ~is_right_end
    is_right_end_only = is_critical & is_right_end & ~is_left_end

    # Obtain left and right neighbors
    left_neighbors = critical_block_info[:, CBFields.LEFT_NEIGHBOR]
    right_neighbors = critical_block_info[:, CBFields.RIGHT_NEIGHBOR]

    # Get job indices of operations and their neighbors
    ops_idx = jnp.arange(num_ops_total)
    job_idx = ops_idx // max_num_ops
    right_job_idx = right_neighbors // max_num_ops
    left_job_idx = left_neighbors // max_num_ops

    # Check if the left/right end operation can be moved right or left
    # It can be moved if it is not of the same job as the right/left neighbor respectively
    is_move_right_valid = jnp.logical_and(is_left_end_only, job_idx != right_job_idx)
    is_move_left_valid = jnp.logical_and(is_right_end_only, job_idx != left_job_idx)

    action_mask = jnp.zeros((num_ops_total, 2), dtype=jnp.bool_)
    action_mask = action_mask.at[:, 1].set(is_move_right_valid)
    action_mask = action_mask.at[:, 0].set(is_move_left_valid)

    return action_mask


def _extract_n6_basic_info(
    critical_block_info: chex.Array,
    max_num_ops: int,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Extract basic operation and critical block information.

    Args:
        critical_block_info: Critical block information array
        max_num_ops: Maximum operations per job

    Returns:
        Tuple of (ops_idx, ops_job_ids, ops_order_in_job, is_critical, block_ids, is_left, is_right)
    """
    num_ops_total = critical_block_info.shape[0]
    ops_idx = jnp.arange(num_ops_total)
    ops_job_ids = ops_idx // max_num_ops
    ops_order_in_job = ops_idx % max_num_ops
    is_critical = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH].astype(jnp.bool_)
    block_ids = critical_block_info[:, CBFields.BLOCK_ID]
    is_left = critical_block_info[:, CBFields.IS_LEFT].astype(jnp.bool_)
    is_right = critical_block_info[:, CBFields.IS_RIGHT].astype(jnp.bool_)

    return ops_idx, ops_job_ids, ops_order_in_job, is_critical, block_ids, is_left, is_right


def _compute_job_precedence_constraints(
    ops_idx: chex.Array,
    ops_job_ids: chex.Array,
    block_ids: chex.Array,
    max_num_ops: int,
) -> Tuple[chex.Array, chex.Array]:
    """Compute job precedence constraints for N6 moves.

    Operations can only move to positions where they respect job precedence.

    Args:
        ops_idx: Operation indices
        ops_job_ids: Job IDs for each operation
        block_ids: Critical block IDs
        max_num_ops: Maximum operations per job

    Returns:
        Tuple of (job_precedence_left_respected, job_precedence_right_respected)
    """
    num_ops_total = len(ops_idx)
    num_jobs = num_ops_total // max_num_ops
    num_segments = num_jobs * num_ops_total
    group_id = ops_job_ids * num_ops_total + block_ids

    # Find first/last operations grouped by (job, critical_block)
    first_op_idx_in_group = jax.ops.segment_min(ops_idx, group_id, num_segments=num_segments)
    last_op_idx_in_group = jax.ops.segment_max(ops_idx, group_id, num_segments=num_segments)

    # Check job precedence for each operation
    # Job precedence for left move is respected if the operation is the first in its group
    # Job precedence for right move is respected if the operation is the last in its group
    first_op_for_this_op_group = first_op_idx_in_group[group_id]
    last_op_for_this_op_group = last_op_idx_in_group[group_id]
    job_precedence_left_respected = ops_idx == first_op_for_this_op_group
    job_precedence_right_respected = ops_idx == last_op_for_this_op_group

    return job_precedence_left_respected, job_precedence_right_respected


def _compute_physical_movement_constraints(
    is_critical: chex.Array,
    is_left: chex.Array,
    is_right: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Compute physical movement constraints based on operation positions.

    Args:
        is_critical: Boolean mask of critical operations
        is_left: Boolean mask of left-end operations
        is_right: Boolean mask of right-end operations

    Returns:
        Tuple of (is_internal, is_left_end_only, is_right_end_only, can_physically_move_left/right)
    """
    # Operations can only move if they're not already at the target position
    can_physically_move_left = ~is_left  # Can move left if not already leftmost
    can_physically_move_right = ~is_right  # Can move right if not already rightmost

    # Classify operations by their position in critical blocks
    is_internal = is_critical & can_physically_move_left & can_physically_move_right
    is_left_end_only = is_critical & ~can_physically_move_left & can_physically_move_right
    is_right_end_only = is_critical & can_physically_move_left & ~can_physically_move_right

    return (
        is_internal,
        is_left_end_only,
        is_right_end_only,
        (can_physically_move_left, can_physically_move_right),
    )


def _check_right_move_acyclic_constraints(
    critical_block_info: chex.Array,
    est_ops: chex.Array,
    ops_idx: chex.Array,
    ops_order_in_job: chex.Array,
    ops_job_ids: chex.Array,
    num_ops_per_job: chex.Array,
    max_num_ops: int,
) -> chex.Array:
    """Check that right moves maintain an acyclic graph.

    Args:
        critical_block_info: Critical block information
        est_ops: Earliest start times (without source/target nodes)
        ops_idx: Operation indices
        ops_order_in_job: Order of operations within their jobs
        ops_job_ids: Job IDs for each operation
        num_ops_per_job: Number of operations per job
        max_num_ops: Maximum operations per job

    Returns:
        Boolean array indicating which operations can move right without creating cycles
    """
    right_end_idx = critical_block_info[:, CBFields.RIGHT_END]

    # Obtain earliest start time of job predecessor of right end operation
    has_right_end_job_predecessor = right_end_idx % max_num_ops == 0
    job_predecessor_of_right_end_idx = right_end_idx - 1
    est_job_predecessor_of_right_end_idx = est_ops[job_predecessor_of_right_end_idx]

    # Obtain earliest start time of job successor of current operation
    is_current_op_last_of_job = ops_order_in_job == num_ops_per_job[ops_job_ids] - 1
    job_successor_of_current_op_idx = jnp.where(is_current_op_last_of_job, -1, ops_idx + 1)
    est_job_successor_of_current_op = est_ops[job_successor_of_current_op_idx]

    # If current op's job successor starts after right_end's job predecessor,
    # moving current op to right_end position won't create an invalid schedule
    job_predecessor_condition_satisfied = jnp.where(
        has_right_end_job_predecessor | is_current_op_last_of_job,
        True,
        est_job_successor_of_current_op > est_job_predecessor_of_right_end_idx,
    )

    # If current op's job successor starts after right_end's machine predecessor,
    # moving current op to right_end position won't create an invalid schedule
    machine_predecessor_of_right_end_idx = critical_block_info[
        right_end_idx, CBFields.LEFT_NEIGHBOR
    ]
    est_machine_predecessor_of_right_end_idx = est_ops[machine_predecessor_of_right_end_idx]
    machine_predecessor_condition_satisfied = (
        est_machine_predecessor_of_right_end_idx <= est_job_successor_of_current_op
    ) | is_current_op_last_of_job

    return job_predecessor_condition_satisfied & machine_predecessor_condition_satisfied


def _check_left_move_temporal_constraints(
    critical_block_info: chex.Array,
    est_ops: chex.Array,
    ops_idx: chex.Array,
    ops_order_in_job: chex.Array,
    num_ops_per_job: chex.Array,
    max_num_ops: int,
) -> chex.Array:
    """Check temporal precedence constraints for left moves.

    Args:
        critical_block_info: Critical block information
        est_ops: Earliest start times (without source/target nodes)
        ops_idx: Operation indices
        ops_order_in_job: Order of operations within their jobs
        num_ops_per_job: Number of operations per job
        max_num_ops: Maximum operations per job

    Returns:
        Boolean array indicating which operations can move left without violating constraints
    """
    left_end_idx = critical_block_info[:, CBFields.LEFT_END]

    # Obtain earliest start time of job successor of left end operation
    is_left_end_last_of_job = (
        left_end_idx % max_num_ops == num_ops_per_job[left_end_idx // max_num_ops] - 1
    )
    job_successor_of_left_end_idx = jnp.where(is_left_end_last_of_job, -1, left_end_idx + 1)
    est_job_successor_of_left_end_idx = est_ops[job_successor_of_left_end_idx]

    # Obtain earliest start time of job predecessor of current operation
    is_current_op_first_of_job = ops_order_in_job == 0
    job_predecessor_of_current_op_idx = ops_idx - 1
    est_job_predecessor_of_current_op_idx = est_ops[job_predecessor_of_current_op_idx]

    # If current op's job predecessor starts before left_end's job successor,
    # moving current op to left_end position won't create an invalid schedule
    job_successor_condition_satisfied = jnp.where(
        is_current_op_first_of_job | is_left_end_last_of_job,
        True,
        est_job_predecessor_of_current_op_idx < est_job_successor_of_left_end_idx,
    )

    # If current op's job predecessor starts before left_end's machine successor,
    # moving current op to left_end position won't create an invalid schedule
    machine_successor_of_left_end_idx = critical_block_info[left_end_idx, CBFields.RIGHT_NEIGHBOR]
    est_machine_successor_of_left_end_idx = est_ops[machine_successor_of_left_end_idx]
    machine_successor_condition_satisfied = (
        est_job_predecessor_of_current_op_idx <= est_machine_successor_of_left_end_idx
    ) | is_current_op_first_of_job

    return job_successor_condition_satisfied & machine_successor_condition_satisfied


def _handle_two_operation_blocks_special_case(
    critical_block_info: chex.Array,
) -> chex.Array:
    """Handle special case for two-operation critical blocks.

    When a critical block has only 2 operations, moving left-to-right and right-to-left
    is equivalent. We only allow the left operation to move to avoid duplicate actions.

    Args:
        critical_block_info: Critical block information

    Returns:
        Boolean array indicating which blocks have only 2 operations
    """
    return (
        critical_block_info[:, CBFields.LEFT_NEIGHBOR] == critical_block_info[:, CBFields.LEFT_END]
    )


def _compute_final_n6_action_mask(
    is_internal: chex.Array,
    is_left_end_only: chex.Array,
    is_right_end_only: chex.Array,
    job_precedence_left_respected: chex.Array,
    job_precedence_right_respected: chex.Array,
    no_cycle_created_with_left_move: chex.Array,
    no_cycle_created_with_right_move: chex.Array,
    has_cb_only_2_ops: chex.Array,
) -> chex.Array:
    """Compute the final action mask by combining all constraints.

    Args:
        is_internal: Boolean mask of internal operations
        is_left_end_only: Boolean mask of left-end-only operations
        is_right_end_only: Boolean mask of right-end-only operations
        job_precedence_left_respected: Boolean mask for left job precedence
        job_precedence_right_respected: Boolean mask for right job precedence
        no_cycle_created_with_left_move: Boolean mask for left cycle constraints
        no_cycle_created_with_right_move: Boolean mask for right cycle constraints
        has_cb_only_2_ops: Boolean mask for two-operation blocks

    Returns:
        Action mask of shape (num_ops_total, 2)
    """
    # Left moves: valid for internal operations and right-end operations
    left_move_valid = (
        (is_internal | (is_right_end_only & ~has_cb_only_2_ops))
        & job_precedence_left_respected
        & no_cycle_created_with_left_move
    )

    # Right moves: valid for internal operations and left-end operations
    right_move_valid = (
        (is_internal | is_left_end_only)
        & job_precedence_right_respected
        & no_cycle_created_with_right_move
    )

    # Stack left and right validity into final action mask
    return jnp.stack([left_move_valid, right_move_valid], axis=-1).astype(jnp.bool_)


def get_action_mask_n6(
    critical_block_info: chex.Array,
    max_num_ops: int,
    est: chex.Array,
    num_ops_per_job: chex.Array,
) -> chex.Array:
    """
    Get the mask of valid actions for the N6 neighborhood.

    The N6 neighborhood allows any operation within a critical block to be moved
    to the beginning or end of the critical block, subject to job precedence constraints.

    Documented in [doc.md](doc.md#n6-neighborhood).

    Args:
        critical_block_info: Padded array of critical block info, shape (num_ops_total, 8)
        max_num_ops: Max operations per job (will be a static argument)
        est: Array of earliest start times, shape (max_num_jobs * max_num_ops + 2,)
        num_ops_per_job: Array containing the number of operations per job

    Returns:
        Action mask of shape (num_ops_total, 2) where [:, 0] is left moves and [:, 1] is right moves
    """
    est_ops = est[1:-1]  # remove source and target nodes

    # 1. Extract basic operation and critical block information
    ops_idx, ops_job_ids, ops_order_in_job, is_critical, block_ids, is_left, is_right = (
        _extract_n6_basic_info(critical_block_info, max_num_ops)
    )

    # 2. Compute job precedence constraints
    job_precedence_left_respected, job_precedence_right_respected = (
        _compute_job_precedence_constraints(ops_idx, ops_job_ids, block_ids, max_num_ops)
    )

    # 3. Compute physical movement constraints
    is_internal, is_left_end_only, is_right_end_only, _ = _compute_physical_movement_constraints(
        is_critical, is_left, is_right
    )

    # 4. Check right move acyclic graph constraints
    no_cycle_created_with_right_move = _check_right_move_acyclic_constraints(
        critical_block_info,
        est_ops,
        ops_idx,
        ops_order_in_job,
        ops_job_ids,
        num_ops_per_job,
        max_num_ops,
    )

    # 5. Check left move temporal precedence constraints
    no_cycle_created_with_left_move = _check_left_move_temporal_constraints(
        critical_block_info, est_ops, ops_idx, ops_order_in_job, num_ops_per_job, max_num_ops
    )

    # 6. Handle special case for two-operation critical blocks
    has_cb_only_2_ops = _handle_two_operation_blocks_special_case(critical_block_info)

    # 7. Compute final action mask
    return _compute_final_n6_action_mask(
        is_internal,
        is_left_end_only,
        is_right_end_only,
        job_precedence_left_respected,
        job_precedence_right_respected,
        no_cycle_created_with_left_move,
        no_cycle_created_with_right_move,
        has_cb_only_2_ops,
    )


def select_operations_to_switch(
    critical_block_info: chex.Array, chosen_action: chex.Array, neighborhood: Neighborhood
) -> chex.Array:
    """Convert a chosen action into the pair of operations to switch in the schedule.
    This function determines which two operations need to be swapped based on the chosen
    operation and movement direction.

    Direction effects:
    - direction=0 (left): The chosen operation moves earlier in the schedule by swapping
      with its left neighbor (N5) or left end of block (N6)
    - direction=1 (right): The chosen operation moves later in the schedule by swapping
      with its right neighbor (N5) or right end of block (N6)
    NB: direction is redundant for N5 neighborhood but necessary for N6.

    Args:
        critical_block_info: array of critical block information, shape (num_ops_total, 8)
        chosen_action: array of chosen action, shape (2,) containing:
            - chosen_op_idx: index of the chosen operation to move
            - chosen_left_or_right: 0 to move left, 1 to move right
        neighborhood: Neighborhood.N5 for N5 (adjacent swaps), Neighborhood.N6 for
        N6 (block end swaps)

    Returns:
        Array of shape (3,) containing (start_op_idx, end_op_idx, direction) where
        start_op_idx is executed before end_op_idx in the current schedule order.
    """

    chosen_op_idx, left_or_right = chosen_action

    # Determine the index of the operation to swap with:
    # - For N5, this is the immediate left or right neighbor in the critical block.
    # - For N6, this is the left_end or right_end of the critical block.
    neighbor_idx = jnp.where(
        neighborhood == Neighborhood.N5,
        critical_block_info[chosen_op_idx, CBFields.LEFT_NEIGHBOR + left_or_right],
        critical_block_info[chosen_op_idx, CBFields.LEFT_END + left_or_right],
    )

    # Return the operation pair in schedule order:
    # - If moving left (left_or_right == 0), neighbor comes before chosen_op_idx.
    # - If moving right (left_or_right == 1), chosen_op_idx comes before neighbor.
    # The third element indicates the direction.
    return jnp.where(
        left_or_right == 0,
        jnp.array(
            [neighbor_idx, chosen_op_idx, 0]
        ),  # 0: neighbor remains, chosen_op_idx moves before neighbor
        jnp.array(
            [chosen_op_idx, neighbor_idx, 1]
        ),  # 1: chosen_op_idx remains, neighbor moves after chosen_op_idx
    )

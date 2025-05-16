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


def get_critical_operations(
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

    Args:
        est: array of earliest start times, shape (max_num_jobs * max_num_ops + 2,) including
             source and target nodes
        lst: array of latest start times, shape (max_num_jobs * max_num_ops + 2,) including
             source and target nodes
        adj_mat_mc: adjacency matrix of the machine constraint graph,
                    shape (max_num_jobs * max_num_ops + 2, max_num_jobs * max_num_ops + 2)
        ops_durations: array of operation durations, shape (max_num_jobs * max_num_ops,)

    Returns:
        Array of indices of critical operations of shape (num_ops_total, 8) containing :
        - is_on_critical_path : 1 if the operation is on the critical path, 0 otherwise
        - num_critical_block : the number of the critical block the operation belongs to
        - is_left : 1 if the operation is a left operation of a critical block, 0 otherwise
        - is_right : 1 if the operation is a right operation of a critical block, 0 otherwise
        - left_neighbor : the left neighbor of the operation
        - right_neighbor : the right neighbor of the operation
        - left_end : operation index of the left end of the critical block
        - right_end : operation index of the right end of the critical block
    """
    num_ops_total = max_num_ops * max_num_jobs

    # Drop source and target nodes
    est = est[1:-1]
    lst = lst[1:-1]

    # Find operations where est == lst
    critical_mask = jnp.isclose(est, lst)

    # Get machine constraints adjacency matrix without source/target nodes
    adj_mat_mc_ops = adj_mat_mc[1:-1, 1:-1]

    ops_durations = ops_durations.reshape(-1)  # flatten ops_durations

    # Create adjacency matrix of critical operations
    senders, receivers = jnp.nonzero(
        adj_mat_mc_ops > 0, size=max_num_edges, fill_value=-1
    )  # shape (max_num_edges,) for senders and receivers

    # Create a mask to filter out dummy edges (where indices are -1)
    valid_pairs = jnp.logical_and(senders != -1, receivers != -1)

    # Replace invalid indices with a safe default (e.g., 0) to avoid out-of-bounds access
    # These will be ignored later thanks to the valid_pairs mask
    safe_senders = jnp.where(valid_pairs, senders, 0)
    safe_receivers = jnp.where(valid_pairs, receivers, 0)

    # Check whether both operations are marked as critical
    is_critical_pair = jnp.logical_and(critical_mask[safe_senders], critical_mask[safe_receivers])

    # Check if the operations are adjacent in time (finish of one == start of the next)
    is_adjacent = jnp.isclose(est[safe_senders] + ops_durations[safe_senders], est[safe_receivers])

    # A pair is part of a critical block if:
    # (1) both ops are valid (not padded),
    # (2) both are on the critical path,
    # (3) they are adjacent in time.
    critical_blocks_mask = jnp.logical_and(
        valid_pairs, jnp.logical_and(is_critical_pair, is_adjacent)
    )  # shape (max_num_edges,)
    senders = jnp.where(critical_blocks_mask, senders, -1)
    receivers = jnp.where(critical_blocks_mask, receivers, -1)

    # Sort pairs by earliest start time, with +inf for non-critical pairs
    filtered_est = jnp.where(critical_blocks_mask, est[senders], jnp.inf)  # shape (max_num_edges,)
    sorted_est = jnp.argsort(filtered_est)
    critical_block_pairs = jnp.stack(
        [senders[sorted_est], receivers[sorted_est]], axis=-1
    )  # shape (max_num_edges, 2)

    # === Initialize critical_block_info ===

    # critical_block_info is a matrix of shape (num_ops_total, 8) containing:
    # (is_on_critical_path, num_critical_block, is_left, is_right, left_neighbor, right_neighbor,
    # left_end, right_end)

    critical_block_info = jnp.zeros((num_ops_total, 8), dtype=jnp.int32)

    # Set is_on_critical_path
    critical_block_info = critical_block_info.at[:, 0].set(critical_mask)
    # Each operation initially defines its own critical block,
    # set up num_critical_block and left_end accordingly (updated during scan)
    critical_block_info = critical_block_info.at[:, 1].set(jnp.arange(num_ops_total))
    critical_block_info = critical_block_info.at[:, 6].set(jnp.arange(num_ops_total))
    # All ops are initially marked as right ends (updated during scan)
    critical_block_info = critical_block_info.at[:, 3].set(1)
    # Initialize neighbors and right_end to -1
    critical_block_info = critical_block_info.at[:, 4:6].set(-1)
    critical_block_info = critical_block_info.at[:, 7].set(-1)

    # Initialize right_end_array to -1
    # Right end array is used to
    right_end_array = jnp.full(num_ops_total, -1, dtype=jnp.int32)

    def update_critical_block_with_pair(
        carry: Tuple[chex.Array, chex.Array], critical_block_pair: Tuple[jnp.int32, jnp.int32]
    ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
        """Update the critical block information based on the critical block pair.
           During one update, if the pair (start_idx, end_idx) is valid, we update the
           critical block information as follows:
           - num_critical_block of end_idx is set to the block_id of the start_idx operation.
           - left_end of end_idx is set to left_end of start_idx.
           - is_right of start_idx is set to 0.
           - right_neighbor of start_idx is set to the end_idx operation.
           - is_left of end_idx is set to 0.
           - left_neighbor of end_idx is set to the start_idx operation.

        Args:
            carry: Tuple containing (critical_block_info, right_end_array)
            critical_block_pair: Tuple containing (start_idx, end_idx)

        Returns:
            Tuple containing (updated critical_block_info, updated right_end_array)
        """

        start_idx, end_idx = critical_block_pair

        valid_condition = jnp.logical_and(start_idx != -1, end_idx != -1)

        def if_valid_pair(
            carry: Tuple[chex.Array, chex.Array], critical_block_pair: Tuple[jnp.int32, jnp.int32]
        ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
            critical_block_info, right_end_array = carry
            start_idx, end_idx = critical_block_pair

            block_id = critical_block_info[start_idx, 1]
            left_end_of_start_idx = critical_block_info[start_idx, 6]

            critical_block_info = (
                critical_block_info.at[end_idx, 1]
                .set(block_id)  # num_critical_block
                .at[start_idx, 3]
                .set(0)  # is_right = 0 for start_idx
                .at[start_idx, 5]
                .set(end_idx)  # right_neighbor for start_idx
                .at[end_idx, 4]
                .set(start_idx)  # left_neighbor for end_idx
                .at[end_idx, 6]
                .set(left_end_of_start_idx)  # left_end for end_idx
            )

            # set current right_end of the critical block
            right_end_array = right_end_array.at[block_id].set(end_idx)

            return (critical_block_info, right_end_array), None

        def if_invalid_pair(
            carry: Tuple[chex.Array, chex.Array], critical_block_pair: Tuple[jnp.int32, jnp.int32]
        ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
            return carry, None

        new_state, _ = jax.lax.cond(
            valid_condition, if_valid_pair, if_invalid_pair, carry, critical_block_pair
        )
        return new_state, None

    # Initial state = (critical_block_info, right_end_array)
    (final_critical_block_info, right_end_array), _ = jax.lax.scan(
        update_critical_block_with_pair,
        (critical_block_info, right_end_array),
        critical_block_pairs,
    )

    # Set right_end of each operation to the right_end of its critical block
    block_ids = final_critical_block_info[:, 1]
    right_ends = jnp.where(block_ids != -1, right_end_array[block_ids], -1)
    final_critical_block_info = final_critical_block_info.at[:, 7].set(right_ends)

    # For each critical block, find left operations.
    # Due to the initialization of num_critical_block, the left operation are the ones
    # with index equal to num_critical_block.
    final_critical_block_info = final_critical_block_info.at[:, 2].set(
        final_critical_block_info[:, 1] == jnp.arange(num_ops_total)
    )

    return final_critical_block_info


def get_action_mask_n5(critical_block_info: chex.Array, max_num_ops: int) -> chex.Array:
    """Get the mask of valid actions for the N5 neighborhood.

    Args:
        critical_block_info: array of critical block information, shape (num_ops_total, 8)

    Returns:
        A mask of valid actions, shape (num_ops_total, 2)
    """
    num_ops_total = critical_block_info.shape[0]

    # Extract left and right ends of critical blocks
    is_critical = critical_block_info[:, 0]
    is_left_end = critical_block_info[:, 2]
    is_right_end = critical_block_info[:, 3]

    # Mask operations that are both left and right ends
    # (1 operation per critical block, no possible action)
    is_left_end_only = is_critical & is_left_end & ~is_right_end
    is_right_end_only = is_critical & is_right_end & ~is_left_end

    # Get left and right neighbors
    left_neighbors = critical_block_info[:, 4]
    right_neighbors = critical_block_info[:, 5]

    # Get job indices of operations and their neighbors
    ops_idx = jnp.arange(num_ops_total)
    job_idx = ops_idx // max_num_ops
    right_job_idx = right_neighbors // max_num_ops
    left_job_idx = left_neighbors // max_num_ops

    # Get valid N5 operations
    is_move_right_valid = jnp.logical_and(is_left_end_only, job_idx != right_job_idx)
    is_move_left_valid = jnp.logical_and(is_right_end_only, job_idx != left_job_idx)

    action_mask = jnp.zeros((num_ops_total, 2), dtype=jnp.bool_)
    action_mask = action_mask.at[:, 1].set(is_move_right_valid)
    action_mask = action_mask.at[:, 0].set(is_move_left_valid)

    return action_mask


def select_operations_to_switch(
    critical_block_info: chex.Array, chosen_action: chex.Array, neighborhood: int
) -> chex.Array:
    """From the chosen operation index and the action to move it left or right, transform it
    into the pair of operations to switch. By convention, the pair of operations (start, end)
    to switch has to be in the order start < end in the schedule.

    Args:
        critical_block_info: array of critical block information, shape (num_ops_total, 6)
        chosen_action: array of chosen action, shape (2,) and contains:
        - chosen_op_idx: index of the chosen operation
        - chosen_left_or_right: 0 to move the operation left, 1 to move it right
        neighborhood: 5 for N5, 6 for N6.

    Returns: a pair of operations (i, j) that have to be switched.
    """

    chosen_op_idx, left_or_right = chosen_action

    # Get the index of the other operation to switch
    # If the neighborhood is N5, it is necessarily the directly left/right operation
    # of chosen_op_idx in the critical block.
    # If the neighborhood is N6, it is necessarily the left_end/right_end operation
    # of the critical block.
    neighbor_idx = jnp.where(
        neighborhood == 5,
        critical_block_info[chosen_op_idx, 4 + left_or_right],
        critical_block_info[chosen_op_idx, 6 + left_or_right],
    )

    # If left neighbor, return [neighbor_idx, chosen_op_idx] to maintain i before j convention
    # If right neighbor, return [chosen_op_idx, neighbor_idx] since
    # chosen_op_idx is already before neighbor_idx
    return jnp.where(
        left_or_right == 0,
        jnp.array(
            [neighbor_idx, chosen_op_idx, 0]
        ),  # 0 = start stays at its position and end moves before it
        jnp.array(
            [chosen_op_idx, neighbor_idx, 1]
        ),  # 1 = end stays at its position and start moves after it
    )

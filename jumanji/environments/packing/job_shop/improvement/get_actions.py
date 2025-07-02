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
        Array of indices of critical operations of shape (num_ops_total, 8), see CBFields.
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
    critical_block_info = critical_block_info.at[:, CBFields.IS_ON_CRITICAL_PATH].set(critical_mask)
    # Each operation initially defines its own critical block,
    # set up num_critical_block and left_end accordingly (updated during scan)
    critical_block_info = critical_block_info.at[:, CBFields.BLOCK_ID].set(
        jnp.arange(num_ops_total)
    )
    critical_block_info = critical_block_info.at[:, CBFields.LEFT_END].set(
        jnp.arange(num_ops_total)
    )
    # All ops are initially marked as right ends (updated during scan)
    critical_block_info = critical_block_info.at[:, CBFields.IS_RIGHT].set(1)
    # Initialize neighbors and right_end to -1
    critical_block_info = critical_block_info.at[:, CBFields.LEFT_NEIGHBOR].set(-1)
    critical_block_info = critical_block_info.at[:, CBFields.RIGHT_NEIGHBOR].set(-1)
    critical_block_info = critical_block_info.at[:, CBFields.RIGHT_END].set(-1)

    # Initialize right_end_array to -1
    # Right end array is used to
    right_end_array = jnp.full(num_ops_total, -1, dtype=jnp.int32)

    def cond_fun(loop_state: Tuple[jnp.int32, Tuple[chex.Array, chex.Array]]) -> jnp.bool_:
        """Condition function for the while loop.
        The iteration stops when we have processed all the valid critical block pairs (i.e.
        while start_idx and end_idx are both not -1).
        """
        i, _ = loop_state
        start_idx, end_idx = critical_block_pairs[i]
        return jnp.logical_and(i < num_ops_total, jnp.logical_and(start_idx != -1, end_idx != -1))

    def body_fun(
        loop_state: Tuple[jnp.int32, Tuple[chex.Array, chex.Array]],
    ) -> Tuple[jnp.int32, Tuple[chex.Array, chex.Array]]:
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
            loop_state: Tuple containing (i, (critical_block_info, right_end_array))
                   with i being the current index in the critical_block_pairs array

        Returns:
            Tuple containing (updated critical_block_info, updated right_end_array)
        """
        i, (critical_block_info, right_end_array) = loop_state
        start_idx, end_idx = critical_block_pairs[i]

        block_id = critical_block_info[start_idx, 1]
        left_end_of_start_idx = critical_block_info[start_idx, 6]

        critical_block_info = (
            critical_block_info.at[end_idx, CBFields.BLOCK_ID]
            .set(block_id)
            .at[start_idx, CBFields.IS_RIGHT]
            .set(0)
            .at[start_idx, CBFields.RIGHT_NEIGHBOR]
            .set(end_idx)
            .at[end_idx, CBFields.LEFT_NEIGHBOR]
            .set(start_idx)
            .at[end_idx, CBFields.LEFT_END]
            .set(left_end_of_start_idx)
        )

        right_end_array = right_end_array.at[block_id].set(end_idx)

        return (i + 1, (critical_block_info, right_end_array))

    init_state = (0, (critical_block_info, right_end_array))

    _, (final_critical_block_info, right_end_array) = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # Set right_end of each operation to the right_end of its critical block
    block_ids = final_critical_block_info[:, 1]
    right_ends = jnp.where(block_ids != -1, right_end_array[block_ids], -1)
    final_critical_block_info = final_critical_block_info.at[:, CBFields.RIGHT_END].set(right_ends)

    # For each critical block, find left operations.
    # Due to the initialization of num_critical_block, the left operation are the ones
    # with index equal to num_critical_block.
    final_critical_block_info = final_critical_block_info.at[:, CBFields.IS_LEFT].set(
        final_critical_block_info[:, CBFields.BLOCK_ID] == jnp.arange(num_ops_total)
    )

    return final_critical_block_info


def get_critical_operations_plus_empty_space_left_right(
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
        Array of indices of critical operations of shape (num_ops_total, 8), see CBFields.
    """
    num_ops_total = max_num_ops * max_num_jobs

    # Drop source and target nodes
    makespan = est[-1]
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

    #add dummy last operation
    safe_senders_dummy_last = jnp.where(valid_pairs, senders, -1)
    safe_receivers_dummy_last = jnp.where(valid_pairs, receivers, -1)

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
    critical_block_info = critical_block_info.at[:, CBFields.IS_ON_CRITICAL_PATH].set(critical_mask)
    # Each operation initially defines its own critical block,
    # set up num_critical_block and left_end accordingly (updated during scan)
    critical_block_info = critical_block_info.at[:, CBFields.BLOCK_ID].set(
        jnp.arange(num_ops_total)
    )
    critical_block_info = critical_block_info.at[:, CBFields.LEFT_END].set(
        jnp.arange(num_ops_total)
    )
    # All ops are initially marked as right ends (updated during scan)
    critical_block_info = critical_block_info.at[:, CBFields.IS_RIGHT].set(1)
    # Initialize neighbors and right_end to -1
    critical_block_info = critical_block_info.at[:, CBFields.LEFT_NEIGHBOR].set(-1)
    critical_block_info = critical_block_info.at[:, CBFields.RIGHT_NEIGHBOR].set(-1)
    critical_block_info = critical_block_info.at[:, CBFields.RIGHT_END].set(-1)

    # Initialize right_end_array to -1
    # Right end array is used to
    right_end_array = jnp.full(num_ops_total, -1, dtype=jnp.int32)

    def cond_fun(loop_state: Tuple[jnp.int32, Tuple[chex.Array, chex.Array]]) -> jnp.bool_:
        """Condition function for the while loop.
        The iteration stops when we have processed all the valid critical block pairs (i.e.
        while start_idx and end_idx are both not -1).
        """
        i, _ = loop_state
        start_idx, end_idx = critical_block_pairs[i]
        return jnp.logical_and(i < num_ops_total, jnp.logical_and(start_idx != -1, end_idx != -1))

    def body_fun(
        loop_state: Tuple[jnp.int32, Tuple[chex.Array, chex.Array]],
    ) -> Tuple[jnp.int32, Tuple[chex.Array, chex.Array]]:
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
            loop_state: Tuple containing (i, (critical_block_info, right_end_array))
                   with i being the current index in the critical_block_pairs array

        Returns:
            Tuple containing (updated critical_block_info, updated right_end_array)
        """
        i, (critical_block_info, right_end_array) = loop_state
        start_idx, end_idx = critical_block_pairs[i]

        block_id = critical_block_info[start_idx, 1]
        left_end_of_start_idx = critical_block_info[start_idx, 6]

        critical_block_info = (
            critical_block_info.at[end_idx, CBFields.BLOCK_ID]
            .set(block_id)
            .at[start_idx, CBFields.IS_RIGHT]
            .set(0)
            .at[start_idx, CBFields.RIGHT_NEIGHBOR]
            .set(end_idx)
            .at[end_idx, CBFields.LEFT_NEIGHBOR]
            .set(start_idx)
            .at[end_idx, CBFields.LEFT_END]
            .set(left_end_of_start_idx)
        )

        right_end_array = right_end_array.at[block_id].set(end_idx)

        return (i + 1, (critical_block_info, right_end_array))

    init_state = (0, (critical_block_info, right_end_array))

    _, (final_critical_block_info, right_end_array) = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # Set right_end of each operation to the right_end of its critical block
    block_ids = final_critical_block_info[:, 1]
    right_ends = jnp.where(block_ids != -1, right_end_array[block_ids], -1)
    final_critical_block_info = final_critical_block_info.at[:, CBFields.RIGHT_END].set(right_ends)

    # For each critical block, find left operations.
    # Due to the initialization of num_critical_block, the left operation are the ones
    # with index equal to num_critical_block.
    final_critical_block_info = final_critical_block_info.at[:, CBFields.IS_LEFT].set(
        final_critical_block_info[:, CBFields.BLOCK_ID] == jnp.arange(num_ops_total)
    )

    # AAADDED FOR MORE NODE FEATURES
    # Calcul du gap pour chaque edge machine valide (sender -> receiver)
    end_sender = est[safe_senders_dummy_last] + ops_durations[safe_senders_dummy_last]
    start_receiver = est[safe_receivers_dummy_last]
    gaps = start_receiver - end_sender  # shape (max_num_edges,)

    num_ops = num_ops_total

    # Initialiser arrays gap_left et gap_right avec -1
    gap_left = jnp.full(num_ops, -1.0) #count dummy first operation
    gap_right = jnp.full(num_ops, -1.0) #count dummy first operation

    # Pour chaque receiver, stocker gap avec son sender (gap_left)
    gap_left = gap_left.at[safe_receivers_dummy_last].set(
        gaps
    )
    # Pour chaque sender, stocker gap avec son receiver (gap_right)
    gap_right = gap_right.at[safe_senders_dummy_last].set(
        gaps
    )

    starting_ops = jnp.logical_and(gap_left == -1.0, gap_right != -1.0)
    ending_ops = jnp.logical_and(gap_left != -1.0, gap_right == -1.0)

    #fill gap_left and gap_right with the correct gap
    gap_left = jnp.where(starting_ops, est, gap_left)
    gap_right = jnp.where(ending_ops, makespan - (est + ops_durations), gap_right)

    #remove for non-existing operations (in the test it was -inf)
    gap_left = jnp.where(ops_durations == -1, -1.0, gap_left)
    gap_right = jnp.where(ops_durations == -1, -1.0, gap_right)
    # Concaténer à critical_block_info (en ajoutant 2 colonnes)
    final_critical_block_info = jnp.concatenate(
        [final_critical_block_info, gap_left[:, None], gap_right[:, None]], axis=-1
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
    is_critical = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH].astype(jnp.bool_)
    is_left_end = critical_block_info[:, CBFields.IS_LEFT].astype(jnp.bool_)
    is_right_end = critical_block_info[:, CBFields.IS_RIGHT].astype(jnp.bool_)

    # Mask operations that are both left and right ends
    # (1 operation per critical block, no possible action)
    is_left_end_only = is_critical & is_left_end & ~is_right_end
    is_right_end_only = is_critical & is_right_end & ~is_left_end

    # Get left and right neighbors
    left_neighbors = critical_block_info[:, CBFields.LEFT_NEIGHBOR]
    right_neighbors = critical_block_info[:, CBFields.RIGHT_NEIGHBOR]

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


def get_action_mask_n6(
    critical_block_info: chex.Array,
    max_num_ops: int,
    est: chex.Array,
    num_ops_per_job: int,
) -> chex.Array:
    """
    Get the N6 action mask in a JIT-safe way by using fixed maximum sizes.
    
    Args:
        critical_block_info: Padded array of critical block info.
        num_ops_real: The actual number of valid operations in the padded array.
        max_num_ops: Max operations per job (will be a static argument).
        max_num_jobs: Max jobs in the problem (will be a static argument).
    """
    ### ÉTAPE 2: Utiliser les dimensions statiques, pas .shape ###
    # Calculez la taille totale maximale à partir des arguments statiques.
    num_ops_total = critical_block_info.shape[0]
    
    # === 1. Dériver les IDs sur la base de la taille maximale ===
    ops_idx = jnp.arange(num_ops_total)
    ops_job_ids = ops_idx // max_num_ops

    # === 2. Extraire les informations de base des blocs ===
    is_critical = critical_block_info[:, CBFields.IS_ON_CRITICAL_PATH].astype(jnp.bool_)
    block_ids = critical_block_info[:, CBFields.BLOCK_ID].astype(jnp.int32)

    # === 3. Calculer num_segments de manière 100% STATIQUE ===
    # On utilise SEULEMENT les arguments statiques pour cette valeur critique.
    num_jobs = num_ops_total//max_num_ops
    num_segments = num_jobs * num_ops_total
    
    # L'ID de groupe est calculé pour tout le tableau, y compris le padding.
    group_id = ops_job_ids * num_ops_total + block_ids #block_ids is necessary smaller than num_ops_total, so a group_id matches a unique (job, block) pair
    # jax.debug.print("ops_job_ids: {}", ops_job_ids)
    # jax.debug.print("block_ids: {}", block_ids)
    # jax.debug.print("group_id: {}", group_id)
    # === 4. Trouver les opérations de plus petit et plus grand index dans chaque groupe ===
    # Ces appels sont maintenant JIT-safe car `num_segments` est une constante pour le compilateur.
    first_op_idx_in_group = jax.ops.segment_min(
        ops_idx, group_id, num_segments=num_segments
    )
    last_op_idx_in_group = jax.ops.segment_max(
        ops_idx, group_id, num_segments=num_segments
    )
    # jax.debug.print("first_op_idx_in_group: {}", first_op_idx_in_group)
    # jax.debug.print("last_op_idx_in_group: {}", last_op_idx_in_group)
    # === 5. Vérifier la contrainte de précédence pour chaque opération ===
    first_op_for_this_op_group = first_op_idx_in_group[group_id]
    last_op_for_this_op_group = last_op_idx_in_group[group_id]
    # jax.debug.print("first_op_for_this_op_group: {}", first_op_for_this_op_group)
    # jax.debug.print("last_op_for_this_op_group: {}", last_op_for_this_op_group)

    job_precedence_ok_left = (ops_idx == first_op_for_this_op_group)
    job_precedence_ok_right = (ops_idx == last_op_for_this_op_group)

    # === 6. Combiner toutes les conditions ===
    # 
    is_left = critical_block_info[:, CBFields.IS_LEFT].astype(jnp.bool_)
    is_right = critical_block_info[:, CBFields.IS_RIGHT].astype(jnp.bool_)
    can_physically_move_left = ~is_left
    can_physically_move_right = ~is_right

    is_internal = is_critical & can_physically_move_left & can_physically_move_right  # shape (num_ops_total,)
    is_left_and_not_right = is_critical & ~can_physically_move_left & can_physically_move_right
    is_right_and_not_left = is_critical & can_physically_move_left & ~can_physically_move_right

    # Check that est of job-successor of operation is after est of predecessor of neighbor operation
    #for right move
    right_end_idx = critical_block_info[:, CBFields.RIGHT_END].astype(jnp.int32)
    mask_no_predecessor_right_end = right_end_idx % max_num_ops == 0
    predecessor_of_right_end_idx = right_end_idx - 1 
    est_predecessor_of_right_end_idx = est[predecessor_of_right_end_idx+1] #+1 to have the correct operation index (due to source in est)
    #Get successor of current operation
    mask_no_successor_current_op = ops_idx % max_num_ops == num_ops_per_job[ops_idx//max_num_ops]-1 # relative position to the job =? last one in the job
    successor_op_idx = ops_idx + 1
    successor_op_idx = jnp.where(mask_no_successor_current_op, -1, successor_op_idx)
    est_successor_of_current_op = est[successor_op_idx+1] #+1 to have the correct operation index (due to source in est)
    right_end_precedence_ok = est_successor_of_current_op > est_predecessor_of_right_end_idx
    #Put right_end_precedence ok if there is no successor or predecessor
    right_end_precedence_ok = jnp.where(mask_no_predecessor_right_end | mask_no_successor_current_op, True, right_end_precedence_ok)

    #check also that est of machine predecessor of right end is before est of job successor of current operation
    machine_predecessor_of_right_end_idx = critical_block_info[right_end_idx, CBFields.LEFT_NEIGHBOR]
    est_machine_predecessor_of_right_end_idx = est[machine_predecessor_of_right_end_idx+1] #+1 to have the correct operation index (due to source in est)
    right_end_precedence_ok = right_end_precedence_ok & ((est_machine_predecessor_of_right_end_idx <= est_successor_of_current_op) | mask_no_successor_current_op)

    #for left move
    left_end_idx = critical_block_info[:, CBFields.LEFT_END].astype(jnp.int32)
    mask_no_successor_left_end = left_end_idx % max_num_ops == num_ops_per_job[left_end_idx//max_num_ops]-1 # relative position to the job =? last one in the job
    successor_of_left_end_idx = left_end_idx + 1
    successor_of_left_end_idx = jnp.where(mask_no_successor_left_end, -1, successor_of_left_end_idx)
    est_successor_of_left_end_idx = est[successor_of_left_end_idx+1] #+1 to have the correct operation index (due to source in est)
    mask_no_predecessor_current_op = ops_idx % max_num_ops == 0
    predecessor_of_current_op_idx = ops_idx - 1
    est_predecessor_of_current_op_idx = est[predecessor_of_current_op_idx+1] #+1 to have the correct operation index (due to source in est)
    left_end_precedence_ok = est_predecessor_of_current_op_idx < est_successor_of_left_end_idx
    left_end_precedence_ok = jnp.where(mask_no_predecessor_current_op | mask_no_successor_left_end, True, left_end_precedence_ok)

    #check also that est of machine successor of left end is after est of job predecessor of current operation
    machine_successor_of_left_end_idx = critical_block_info[left_end_idx, CBFields.RIGHT_NEIGHBOR]
    est_machine_successor_of_left_end_idx = est[machine_successor_of_left_end_idx+1] #+1 to have the correct operation index (due to source in est)
    left_end_precedence_ok = left_end_precedence_ok & ((est_predecessor_of_current_op_idx <= est_machine_successor_of_left_end_idx) | mask_no_predecessor_current_op)

    only_2_ops_in_cb_for_right = (critical_block_info[:, CBFields.LEFT_NEIGHBOR]==critical_block_info[:, CBFields.LEFT_END])

    #jax.debug.print("any true in only_2_ops_in_cb_for_right on critical: {}", jnp.any(only_2_ops_in_cb_for_right & is_critical))
    ### ÉTAPE 4: Appliquer le masque de validité au résultat final ###
    # L'action est valide si l'op est VALIDE, critique, peut physiquement bouger,
    # ET respecte la précédence de son job.
    #Asymetric because of the way we define the mask, to have a unique action when there are 2 operations in the critical block
    # (because move left to right and right to left are the same action)
    left_valid = (is_internal | (is_right_and_not_left&~only_2_ops_in_cb_for_right)) & job_precedence_ok_left & left_end_precedence_ok
    right_valid = (is_internal | is_left_and_not_right) & job_precedence_ok_right & right_end_precedence_ok

    # right_valid = is_internal & job_precedence_ok_right
    #CASE WHERE THERE ARE 2 OPERATIONS IN THE CRITICAL BLOCK
    # iN THIS CASE, WE ONLY WANT TO ALLOW MOVE FOR THE LEFT OPERATION

    action_mask = jnp.stack([left_valid, right_valid], axis=-1)

    return action_mask.astype(jnp.bool_)


def select_operations_to_switch(
    critical_block_info: chex.Array, chosen_action: chex.Array, neighborhood: int
) -> chex.Array:
    """From the chosen operation index and the action to move it left or right, transform it
    into the pair of operations to switch. By convention, the pair of operations (start, end)
    to switch has to be in the order start < end in the schedule.

    Args:
        critical_block_info: array of critical block information, shape (num_ops_total, 8)
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
        critical_block_info[chosen_op_idx, CBFields.LEFT_NEIGHBOR + left_or_right],
        critical_block_info[chosen_op_idx, CBFields.LEFT_END + left_or_right],
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

# def fully_convert_to_operation_pairs_N5(
#     critical_block_info: chex.Array,
#     action_mask: chex.Array
#     ) -> chex.Array:
#     """Convert the action mask to the operation pairs mask.
#     Warning: operation_pairs_mask[i, j]== True iff operation i is before j and we can swap them"""
#     num_ops_total = critical_block_info.shape[0]

#     #il faut que je sélectionne les opérations qui ont leur action_mask à true,
#     #et pour celle là je récupère leur index et leur voisin

#     is_move_left_valid = action_mask[:, 0]
#     is_move_right_valid = action_mask[:, 1]

#     #for now, N5 only 
#     left_neighbor_idx = critical_block_info[:, CBFields.LEFT_NEIGHBOR]
#     right_neighbor_idx = critical_block_info[:, CBFields.RIGHT_NEIGHBOR]
#     op_idx = jnp.arange(num_ops_total)

#     #for is_move_left_valid, we want to set the pair (left_neighbor, operation) to True
#     #for is_move_right_valid, we want to set the pair (operation, right_neighbor) to True
#     operation_pairs_mask = jnp.zeros((num_ops_total, num_ops_total), dtype=bool)

#     # Filter non-valid operations pairs
#     #invariant: when replaced by 0, the position in all_mask will be 0 so nothing is changed
#     left_src = jnp.where(is_move_left_valid, left_neighbor_idx, 0)
#     left_dst = jnp.where(is_move_left_valid, op_idx, 0)
#     right_src = jnp.where(is_move_right_valid, op_idx, 0)
#     right_dst = jnp.where(is_move_right_valid, right_neighbor_idx, 0)

#     all_rows = jnp.concatenate([left_src, right_src])
#     all_cols = jnp.concatenate([left_dst, right_dst])
#     all_mask = jnp.concatenate([is_move_left_valid, is_move_right_valid])

#     operation_pairs_mask = operation_pairs_mask.at[all_rows, all_cols].set(all_mask)

#     return operation_pairs_mask


def fully_convert_to_operation_pairs_N5(
    critical_block_info: chex.Array,
    action_mask: chex.Array
    ) -> chex.Array:
    """Convert the action mask to the operation pairs mask.
    operation_pairs_mask[i, j]== True iff operation i is the one concerned by the action (action_mask[i, :] has at least one True)"""
    num_ops_total = critical_block_info.shape[0]

    #il faut que je sélectionne les opérations qui ont leur action_mask à true,
    #et pour celle là je récupère leur index et leur voisin

    is_move_left_valid = action_mask[:, 0]
    is_move_right_valid = action_mask[:, 1]

    #for now, N5 only 
    left_neighbor_idx = critical_block_info[:, CBFields.LEFT_NEIGHBOR]
    right_neighbor_idx = critical_block_info[:, CBFields.RIGHT_NEIGHBOR]
    op_idx = jnp.arange(num_ops_total)

    #for is_move_left_valid, we want to set the pair (operation, left_neighbor) to True
    #for is_move_right_valid, we want to set the pair (operation, right_neighbor) to True
    operation_pairs_mask = jnp.zeros((num_ops_total, num_ops_total), dtype=bool)

    # Filter non-valid operations pairs
    #invariant: when replaced by 0, the position in all_mask will be 0 so nothing is changed
    left_src = jnp.where(is_move_left_valid, op_idx, 0)
    left_dst = jnp.where(is_move_left_valid, left_neighbor_idx, 0)
    right_src = jnp.where(is_move_right_valid, op_idx, 0)
    right_dst = jnp.where(is_move_right_valid, right_neighbor_idx, 0)

    all_rows = jnp.concatenate([left_src, right_src])
    all_cols = jnp.concatenate([left_dst, right_dst])
    all_mask = jnp.concatenate([is_move_left_valid, is_move_right_valid])

    operation_pairs_mask = operation_pairs_mask.at[all_rows, all_cols].set(all_mask)

    return operation_pairs_mask
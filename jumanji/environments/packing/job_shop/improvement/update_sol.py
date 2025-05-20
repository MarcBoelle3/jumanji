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


def link_ops(
    adj_matrix_mc: chex.Array,
    ops_duration: chex.Array,
    op1_mask: chex.Array,  # shape (max_num_jobs * max_num_ops + 2,)
    op2_mask: chex.Array,  # shape (max_num_jobs * max_num_ops + 2,)
) -> chex.Array:
    """Link two operations in the adjacency matrix with a weight equal to the duration of the first op.

    Args:
        adj_matrix_mc: The current machine constraint matrix.
        ops_duration: The duration of the operations (shape (N,)).
        op1: 1-hot mask indicating the start operation to link (or all 0s).
        op2: 1-hot mask indicating the end operation to link (or all 0s).

    Returns:
        The updated machine constraint matrix with the new link.
    """
    start_idx = jnp.argmax(op1_mask)  # At most one 1 in op1
    start_op = start_idx - 1
    end_idx = jnp.argmax(op2_mask)  # At most one 1 in op2
    do_link = jnp.logical_and(jnp.any(op1_mask), jnp.any(op2_mask))

    def update(mat: chex.Array) -> chex.Array:
        start_op_duration = ops_duration.reshape(-1)[start_op]
        return mat.at[start_idx, end_idx].set(start_op_duration)

    adj_matrix_mc = jax.lax.cond(do_link, update, lambda x: x, adj_matrix_mc)

    return adj_matrix_mc


def unlink_ops(
    adj_matrix_mc: chex.Array,
    op1_mask: chex.Array,  # shape (max_num_jobs*max_num_ops+2,)
    op2_mask: chex.Array,  # shape (max_num_jobs*max_num_ops+2,)
) -> chex.Array:
    """Unlink two operations in the adjacency matrix.

    Args:
        adj_matrix_mc: The current machine constraint matrix.
        op1: 1-hot mask indicating the start operation to unlink (or all 0s).
        op2: 1-hot mask indicating the end operation to unlink (or all 0s).

    Returns:
        The updated machine constraint matrix with the link removed.
    """
    i = jnp.argmax(op1_mask)  # at most one 1 in op1
    j = jnp.argmax(op2_mask)  # at most one 1 in op2

    do_unlink = jnp.logical_and(jnp.any(op1_mask), jnp.any(op2_mask))

    def update(mat: chex.Array) -> chex.Array:
        return mat.at[i, j].set(0)

    adj_matrix_mc = jax.lax.cond(do_unlink, update, lambda x: x, adj_matrix_mc)

    return adj_matrix_mc


def get_predecessor(adj_matrix_mc: chex.Array, op: int) -> chex.Array:
    """Get the predecessor mask and weight indicating the predecessor of an operation in the adjacency matrix.

    Args:
        adj_matrix_mc: The current machine constraint matrix.
        op: The operation index to get the predecessor of.
    Returns:
        - The predecessor mask of shape (max_num_jobs*max_num_ops+2,) indicating the predecessor of the operation.
        - The weight of the predecessor.
    """

    return (
        adj_matrix_mc[:, op] != 0
    )  # by construction, there is at most one predecessor; shape (max_num_jobs*max_num_ops+2,)


def get_successor(adj_matrix_mc: chex.Array, op: chex.Array) -> chex.Array:
    """Get the successor mask and weight indicating the successor of an operation in the adjacency matrix.

    Args:
        adj_matrix_mc: The current machine constraint matrix.
        op: The operation index to get the successor of.
    Returns:
        - The successor mask of shape (max_num_jobs*max_num_ops+2,) indicating the successor of the operation.
        - The weight of the successor.
    """

    return (
        adj_matrix_mc[op, :] != 0
    )  # by construction, there is at most one successor; shape (max_num_jobs*max_num_ops+2,)


def update_disjunctive_graph(
    adj_matrix_mc: chex.Array,
    ops_duration: chex.Array,
    action: Tuple[int, int, int],  # à check
) -> chex.Array:
    """Update the machine constraint matrix for a job shop neighborhood move.

    Args:
        adj_matrix_mc: Machine constraint matrix including source and sink.
        ops_duration: Duration of operations.
        action: Tuple (start, end, move_start_end) with operation indices (excluding source).
                move_start_end: 0 = start stays at its position and end moves before it.
                                1 = end stays at its position and start moves after it.
                move_start_end is dummy for N5 neighborhood but required for N6 neighborhood.

    Returns:
        Updated adjacency matrix of the disjunctive graph.
    """
    start, end, move_start_end = action
    start_idx, end_idx = start + 1, end + 1
    n = adj_matrix_mc.shape[0]

    # Create one-hot masks
    start_mask = jnp.arange(n) == start_idx
    end_mask = jnp.arange(n) == end_idx

    # Common nodes
    pred_start = get_predecessor(adj_matrix_mc, op=start_idx)
    succ_start = get_successor(adj_matrix_mc, op=start_idx)
    succ_end = get_successor(adj_matrix_mc, op=end_idx)

    # === Always do: update (start -> succ(end)) and (end -> start) ===
    adj_matrix_mc = unlink_ops(adj_matrix_mc, end_mask, succ_end)
    adj_matrix_mc = link_ops(adj_matrix_mc, ops_duration, start_mask, succ_end)

    adj_matrix_mc = unlink_ops(adj_matrix_mc, start_mask, succ_start)
    adj_matrix_mc = link_ops(adj_matrix_mc, ops_duration, end_mask, start_mask)

    # === Branch: update pred(start) links depending on direction ===
    def case_move_end_to_start(mat: chex.Array) -> chex.Array:
        mat = unlink_ops(mat, pred_start, start_mask)
        mat = link_ops(mat, ops_duration, pred_start, end_mask)
        return mat

    def case_move_start_to_end(mat: chex.Array) -> chex.Array:
        mat = unlink_ops(mat, pred_start, start_mask)
        mat = link_ops(mat, ops_duration, pred_start, succ_start)
        return mat

    adj_matrix_mc = jax.lax.cond(
        move_start_end == 0, case_move_start_to_end, case_move_end_to_start, adj_matrix_mc
    )

    return adj_matrix_mc

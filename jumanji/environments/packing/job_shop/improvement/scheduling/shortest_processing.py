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

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.improvement.scheduling.base import (
    AbstractSchedulingMethod,
)
from jumanji.environments.packing.job_shop.improvement.scheduling.registry import (
    MethodRegistry,
)


@MethodRegistry.register
class ShortestProcessingTimeMethod(AbstractSchedulingMethod):
    """Shortest Processing Time (SPT) scheduling.

    Operations are scheduled in order of increasing processing time.
    At each step, the operation with the shortest duration among available
    candidates is selected.
    """

    NAME = "shortest_processing_time"

    @property
    def name(self) -> str:
        return self.NAME

    def build_machine_adjacency_matrix(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """Build machine adjacency matrix using shortest processing time rule.

        Returns:
            Adjacency matrix (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
            with source/target nodes.
        """
        num_ops_total = self.max_num_jobs * self.max_num_ops

        # Initialize candidate operations (first operation of each job)
        candidate_ops = jnp.arange(0, num_ops_total, self.max_num_ops)

        # Initialize priority ranks (operation durations)
        priority_ranks = ops_durations[:, 0]
        priority_ranks = jnp.where(priority_ranks == -1, jnp.inf, priority_ranks)

        # Initialize adjacency matrix and machine state
        adj_mat = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)
        machine_ops = -jnp.ones((self.num_machines,), dtype=jnp.int32)
        machine_durs = -jnp.ones((self.num_machines,), dtype=jnp.float32)

        def has_candidates(
            carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> jnp.bool_:
            """Check if operations remain to be scheduled."""
            _, _, _, candidates, ranks = carry
            # Only consider candidates with finite ranks (not padding jobs)
            valid_candidates = candidates != -1
            finite_ranks = ranks < jnp.inf
            has_valid_ops = jnp.any(valid_candidates & finite_ranks)
            return has_valid_ops

        def schedule_next_operation(
            carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
            """Schedule the operation with shortest processing time."""
            m_ops, m_durs, adj, candidates, ranks = carry

            # Select job with shortest processing time among valid candidates
            # Mask out invalid candidates (those with infinite ranks or no operations)
            valid_mask = (candidates != -1) & (ranks < jnp.inf)
            masked_ranks = jnp.where(valid_mask, ranks, jnp.inf)
            job_id = jnp.argmin(masked_ranks)
            op_id = candidates[job_id]

            # Get operation details
            op_row, op_col = op_id // self.max_num_ops, op_id % self.max_num_ops
            machine_id = ops_machine_ids[op_row, op_col]
            duration = ops_durations[op_row, op_col]

            # Add machine constraint edge if previous operation exists
            prev_op = m_ops[machine_id]
            prev_dur = m_durs[machine_id]

            adj = jax.lax.cond(
                prev_op != -1,
                lambda a: a.at[prev_op, op_id].set(prev_dur),
                lambda a: a,
                operand=adj,
            )

            # Update machine state
            m_ops = m_ops.at[machine_id].set(op_id)
            m_durs = m_durs.at[machine_id].set(duration)

            # Update candidates and ranks
            num_ops_in_job = jnp.sum(ops_machine_ids[job_id] != -1)
            next_op_id = jnp.where(op_col + 1 < num_ops_in_job, op_id + 1, -1)
            candidates = candidates.at[job_id].set(next_op_id)

            next_rank = jnp.where(
                next_op_id == -1,
                jnp.inf,
                ops_durations[next_op_id // self.max_num_ops, next_op_id % self.max_num_ops],
            )
            ranks = ranks.at[job_id].set(next_rank)

            return m_ops, m_durs, adj, candidates, ranks

        # Schedule all operations
        init_carry = (machine_ops, machine_durs, adj_mat, candidate_ops, priority_ranks)
        _, _, final_adj, _, _ = jax.lax.while_loop(
            has_candidates, schedule_next_operation, init_carry
        )

        # Add source and target padding
        return jnp.pad(final_adj, ((1, 1), (1, 1)), constant_values=0)

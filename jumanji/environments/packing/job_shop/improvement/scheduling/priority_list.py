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
class PriorityListMethod(AbstractSchedulingMethod):
    """Random job priority list scheduling.

    Jobs are scheduled in random order on each machine. The priority order
    is determined by randomly permuting the job indices.
    """

    NAME = "priority_list"

    @property
    def name(self) -> str:
        return self.NAME

    def build_machine_adjacency_matrix(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """Build machine adjacency matrix using random job priority list.

        Returns:
            Adjacency matrix (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
            with source/target nodes.
        """
        if key is None:
            raise ValueError("PriorityListMethod requires a random key")

        # Generate random job priority list using instance attributes
        job_priority = jnp.arange(self.max_num_jobs)
        job_priority = job_priority.at[: self.num_jobs].set(
            jax.random.permutation(key, jnp.arange(self.num_jobs))
        )

        return self._build_adjacency_from_priority(ops_machine_ids, ops_durations, job_priority)

    def _build_adjacency_from_priority(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        job_priority: chex.Array,
    ) -> chex.Array:
        """Build adjacency matrix from job priority ordering."""
        num_ops_total = self.max_num_jobs * self.max_num_ops

        # Reorder operations by job priority
        op_ids = jnp.arange(num_ops_total).reshape((self.max_num_jobs, self.max_num_ops))
        ordered_ops = op_ids[job_priority].reshape(-1)
        ordered_machines = ops_machine_ids[job_priority].reshape(-1)
        ordered_durations = ops_durations[job_priority].reshape(-1)

        # Initialize adjacency matrix
        adj_mat = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)

        # Track last operation on each machine
        init_machine_ops = -jnp.ones((self.num_machines,), dtype=jnp.int32)
        init_machine_durs = -jnp.ones((self.num_machines,), dtype=jnp.float32)

        def add_machine_edge(
            carry: Tuple[chex.Array, chex.Array, chex.Array], x: Tuple[int, float, int]
        ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], None]:
            """Add machine constraint edge."""
            op_id, duration, machine_id = x
            machine_ops, machine_durs, adj = carry

            def update_valid_machine(
                carry_inner: Tuple[chex.Array, chex.Array, chex.Array],
            ) -> Tuple[chex.Array, chex.Array, chex.Array]:
                m_ops, m_durs, adj_inner = carry_inner
                prev_op = m_ops[machine_id]
                prev_dur = m_durs[machine_id]

                # Add edge from previous operation
                adj_inner = jax.lax.cond(
                    prev_op != -1,
                    lambda a: a.at[prev_op, op_id].set(prev_dur),
                    lambda a: a,
                    operand=adj_inner,
                )

                # Update machine state
                m_ops = m_ops.at[machine_id].set(op_id)
                m_durs = m_durs.at[machine_id].set(duration)
                return m_ops, m_durs, adj_inner

            new_carry = jax.lax.cond(
                machine_id != -1, update_valid_machine, lambda c: c, operand=carry
            )
            return new_carry, None

        (_, _, final_adj), _ = jax.lax.scan(
            add_machine_edge,
            (init_machine_ops, init_machine_durs, adj_mat),
            (ordered_ops, ordered_durations, ordered_machines),
        )

        # Add source and target padding
        return jnp.pad(final_adj, ((1, 1), (1, 1)), constant_values=0)

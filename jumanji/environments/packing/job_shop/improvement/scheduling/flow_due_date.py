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

from typing import Optional

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.improvement.scheduling.base import (
    AbstractSchedulingMethod,
)
from jumanji.environments.packing.job_shop.improvement.scheduling.registry import (
    MethodRegistry,
)
from jumanji.environments.packing.job_shop.improvement.scheduling.utils import (
    schedule_op_in_earliest_slot,
)


@MethodRegistry.register
class FlowDueDateMostWorkMethod(AbstractSchedulingMethod):
    """Flow Due Date / Most Work Remaining (FDD/MWR) scheduling.

    Priority is calculated as FDD/MWR ratio, where:
    - FDD: Flow Due Date (cumulative processing time up to current operation)
    - MWR: Most Work Remaining (sum of unscheduled operation durations)

    Lower ratios indicate higher priority.
    """

    NAME = "flow_due_date_most_work"

    @property
    def name(self) -> str:
        return self.NAME

    def build_machine_adjacency_matrix(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """Build machine adjacency matrix using FDD/MWR dispatching rule.

        Returns:
            Adjacency matrix (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
            with source/target nodes.
        """
        num_ops_total = self.max_num_jobs * self.max_num_ops

        # Pre-calculate Flow Due Dates (cumulative durations)
        duration_masked = jnp.where(ops_durations == -1, 0, ops_durations)
        flow_due_dates = jnp.cumsum(duration_masked, axis=1)

        # Run scheduling simulation
        machine_schedule = self._simulate_fdd_mwr_scheduling(
            ops_machine_ids, ops_durations, flow_due_dates
        )

        # Convert schedule to adjacency matrix
        adj_matrix = self._schedule_to_adjacency_matrix(
            machine_schedule, ops_durations, num_ops_total
        )

        # Add source and target padding
        return jnp.pad(adj_matrix, ((1, 1), (1, 1)), constant_values=0)

    def _simulate_fdd_mwr_scheduling(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        flow_due_dates: chex.Array,
    ) -> chex.Array:
        """Simulate FDD/MWR scheduling to determine operation order."""
        num_ops_total = self.max_num_jobs * self.max_num_ops

        # Initialize scheduling state
        machine_schedule = jnp.full((self.num_machines, num_ops_total), -1, dtype=jnp.int32)
        start_times = jnp.zeros_like(machine_schedule, dtype=jnp.float32)
        end_times = jnp.zeros_like(machine_schedule, dtype=jnp.float32)
        job_predecessor_ends = jnp.zeros((self.max_num_jobs,), dtype=jnp.float32)

        # Candidate operations (first of each job)
        candidate_ops = jnp.arange(0, num_ops_total, self.max_num_ops, dtype=jnp.int32)

        # Most Work Remaining (sum of all durations per job)
        duration_masked = jnp.where(ops_durations == -1, 0, ops_durations)
        work_remaining = jnp.sum(duration_masked, axis=1)

        def has_candidates(carry_state: tuple) -> jnp.bool_:
            """Check if any operations remain to be scheduled."""
            _, _, _, _, candidates, work_rem = carry_state
            # Only consider candidates with positive work remaining (not padding jobs)
            valid_candidates = candidates != -1
            valid_work = work_rem > 0
            has_valid_ops = jnp.any(valid_candidates & valid_work)
            return has_valid_ops

        def schedule_next_operation(carry_state: tuple) -> tuple:
            """Schedule operation with best FDD/MWR priority."""
            mch_sched, start_t, end_t, job_ends, candidates, work_rem = carry_state

            # Calculate FDD/MWR priorities
            op_rows = candidates // self.max_num_ops
            op_cols = candidates % self.max_num_ops
            due_dates = flow_due_dates[op_rows, op_cols]

            # Avoid division by zero
            safe_work_rem = jnp.where(work_rem[op_rows] > 0, work_rem[op_rows], 1e-6)
            priorities = due_dates / safe_work_rem

            # Select best operation (lowest priority value) among valid jobs
            # Filter out padding jobs (work_remaining = 0) and invalid candidates
            valid_mask = (candidates != -1) & (work_rem > 0)
            masked_priorities = jnp.where(valid_mask, priorities, jnp.inf)
            best_job_idx = jnp.argmin(masked_priorities)
            op_id = candidates[best_job_idx]

            # Schedule operation in earliest slot
            op_row, op_col = op_id // self.max_num_ops, op_id % self.max_num_ops
            machine_id = ops_machine_ids[op_row, op_col]

            updated_sched, updated_starts, updated_ends, updated_job_ends = (
                schedule_op_in_earliest_slot(
                    op_id,
                    ops_durations,
                    mch_sched[machine_id],
                    start_t[machine_id],
                    end_t[machine_id],
                    job_ends,
                    self.max_num_ops,
                )
            )

            # Update machine schedule
            mch_sched = mch_sched.at[machine_id].set(updated_sched)
            start_t = start_t.at[machine_id].set(updated_starts)
            end_t = end_t.at[machine_id].set(updated_ends)
            job_ends = updated_job_ends

            # Update work remaining
            duration = ops_durations[op_row, op_col]
            work_rem = work_rem.at[best_job_idx].add(-duration)

            # Update candidate operations
            num_ops_in_job = jnp.sum(ops_machine_ids[best_job_idx] != -1)
            next_op_id = jnp.where(op_col + 1 < num_ops_in_job, op_id + 1, -1)
            candidates = candidates.at[best_job_idx].set(next_op_id)

            return mch_sched, start_t, end_t, job_ends, candidates, work_rem

        # Run simulation
        init_carry = (
            machine_schedule,
            start_times,
            end_times,
            job_predecessor_ends,
            candidate_ops,
            work_remaining,
        )
        final_carry = jax.lax.while_loop(has_candidates, schedule_next_operation, init_carry)
        return final_carry[0]

    def _schedule_to_adjacency_matrix(
        self, machine_schedule: chex.Array, ops_durations: chex.Array, num_ops_total: int
    ) -> chex.Array:
        """Convert machine schedule to adjacency matrix."""
        adj_matrix = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)

        def update_machine_links(machine_idx: int, adj: chex.Array) -> chex.Array:
            """Add precedence links for one machine."""
            op_sequence = machine_schedule[machine_idx]
            senders, receivers = op_sequence[:-1], op_sequence[1:]

            # Only link valid operations
            valid_links = (senders != -1) & (receivers != -1)
            safe_senders = jnp.where(valid_links, senders, 0)
            safe_receivers = jnp.where(valid_links, receivers, 0)

            return adj.at[safe_senders, safe_receivers].set(valid_links.astype(jnp.float32))

        # Build adjacency matrix for all machines
        return jax.lax.fori_loop(0, machine_schedule.shape[0], update_machine_links, adj_matrix)

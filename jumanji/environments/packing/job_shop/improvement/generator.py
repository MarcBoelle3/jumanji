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

import abc
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.improvement.compute_makespan import (
    compute_est_lst_makespan,
)
from jumanji.environments.packing.job_shop.improvement.get_actions import (
    get_action_mask_n5,
    get_critical_operations_plus_empty_space_left_right,
    fully_convert_to_operation_pairs_N5,
    get_action_mask_n6
)
from jumanji.environments.packing.job_shop.improvement.types import ImprovementState
from jumanji.environments.packing.job_shop.types import Scenario
from jumanji.environments.packing.job_shop.improvement.initialization_heuristic_rules import schedule_op_in_earliest_slot

class ScheduleGenerator(abc.ABC):
    """Defines the abstract `ScheduleGenerator` base class. A `ScheduleGenerator` is responsible
    for generating an initial solution to a given scenario.
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_jobs: int,
        max_num_ops: int,
    ):
        """Abstract class implementing the attributes `num_jobs`, `num_machines`, `max_num_ops`,
        `max_op_duration`, and `max_num_edges`.

        Args:
            num_jobs: the number of jobs that need to be scheduled.
            num_machines: the number of machines that the jobs can be scheduled on.
            max_num_jobs: the maximum number of jobs that can be scheduled.
                          Static parameter across all instances.
            max_num_ops: the maximum number of operations for any given job.
                         Static parameter across all instances.
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_num_jobs = max_num_jobs
        self.max_num_ops = max_num_ops

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, scenario: Scenario, method_id: int) -> ImprovementState:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.
            scenario: a `Scenario` object that contains the problem instance.
            method_id: the id of the method to use for generating the initial machine constraint
                       adjacency matrix.

        Returns:
            A `JobShopImprovement` environment state.
        """

    def init_adj_mat_pc(self, ops_durations: chex.Array, num_ops_per_job: chex.Array) -> chex.Array:
        """Create the precedence constraint adjacency matrix for the job shop problem.
            Each operation is linked to the next operation of the same job.
            Source node is linked to all starting operations of the jobs.
            All ending operations inside a job are linked to the target node.
            Every edge is weighted by the duration of the starting operation.
            Non-existing operations are masked out with zeros.

        Args:
            ops_durations: Array (max_num_jobs, max_num_ops) indicating the duration
                           of each operation. -1 for padding.
            num_ops_per_job: Array (max_num_jobs,) indicating the number of operations
                            per job. Set to 0 for non-existing jobs.
        Returns:
            The precedence constraint adjacency matrix.
        """

        max_num_jobs, max_num_ops = self.max_num_jobs, self.max_num_ops
        num_ops_total = max_num_jobs * max_num_ops

        ops_durations_flat = ops_durations.reshape(-1)
        ops_mask = ops_durations_flat != -1
        durations = jnp.where(ops_mask, ops_durations_flat, 0)  # shape (num_ops_total,)

        # Base adjacency matrix: link op i to op i+1 with duration
        adj_mat = jnp.diag(durations[:-1], k=1)  # shape (num_ops_total, num_ops_total)

        # Break links between last op of each job and first of next
        job_starts = jnp.arange(0, num_ops_total, max_num_ops)  # shape (max_num_jobs,)
        num_ops_per_job_safe = jnp.maximum(
            num_ops_per_job, 1
        )  # safely handle jobs with no operations
        end_ops_idx = job_starts + num_ops_per_job_safe - 1
        adj_mat = adj_mat.at[end_ops_idx, end_ops_idx + 1].set(0)

        # Mask out columns non-existing operations
        adj_mat = jnp.where(ops_mask[:, None], adj_mat, 0)

        # Pad with source (0) and target (last)
        adj_mat = jnp.pad(adj_mat, ((1, 1), (1, 1)))  # shape (num_ops_total + 2, num_ops_total + 2)

        # Link source (node 0) to first op of each job
        start_indices = 1 + job_starts
        adj_mat = adj_mat.at[0, start_indices].set(1)

        # Link each job's last operation to target
        adj_mat = adj_mat.at[1 + end_ops_idx, -1].set(durations[end_ops_idx])

        # Mask non-existent ops (based on num_jobs)
        valid_op_count = self.num_jobs * max_num_ops
        valid_nodes = jnp.arange(num_ops_total + 2) < (valid_op_count + 1)
        mask = valid_nodes[:, None] & valid_nodes[None, :]

        # Keep target row and column
        mask = mask.at[-1, :].set(True)
        mask = mask.at[:, -1].set(True)
        adj_mat = jnp.where(mask, adj_mat, 0)

        # Remove spurious links from source to non-existent jobs
        adj_mat = adj_mat.at[0].set(jnp.where(valid_nodes, adj_mat[0], 0))

        return adj_mat


class RandomScheduleGenerator(ScheduleGenerator):
    """Schedule generator that initalizes a solution with a random priority list.
    Jobs will be scheduled in the order of the priority list on each machine.
    """

    def __init__(
        self, num_jobs: int, num_machines: int, max_num_jobs: int, max_num_ops: int
    ) -> None:
        super().__init__(num_jobs, num_machines, max_num_jobs, max_num_ops)

    def __call__(self, key: chex.PRNGKey, scenario: Scenario, method_id: int, neighborhood: int) -> ImprovementState:
        # Compute the maximum number of edges in the disjunctive graph.
        max_num_edges = (1 + 2 * self.max_num_ops) * self.max_num_jobs

        ops_machine_ids, ops_durations, num_ops_per_job = (
            scenario.ops_machine_ids,
            scenario.ops_durations,
            scenario.num_ops_per_job,
        )

        # === Initialize adjacency matrices ===

        adj_mat_pc = self.init_adj_mat_pc(ops_durations, num_ops_per_job)
        # Generate random priority list : only used if method_id = 0
        # (adjacency matrix with priority list)
        plist = jnp.arange(self.max_num_jobs)  # shape (max_num_jobs,)
        plist = plist.at[: self.num_jobs].set(
            jax.random.permutation(key, jnp.arange(self.num_jobs))
        )  # shape (max_num_jobs,)
        adj_mat_mc = self.init_adj_mat_mc(method_id, plist, ops_machine_ids, ops_durations)

        # === Compute makespan and scheduled times ===

        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, makespan = compute_est_lst_makespan(adj_mat, ops_durations, max_num_edges)
        scheduled_times = est[1:-1].reshape(
            (self.max_num_jobs, self.max_num_ops)
        )  # discard source and target and set scheduled times to earliest start times

        is_on_critical_path = (est[1:-1] == lst[1:-1]).reshape(
            (self.max_num_jobs, self.max_num_ops)
        )

        # === Compute action mask ===

        # for now: only N5 neighborhood
        critical_block_info, gap_left_right = get_critical_operations_plus_empty_space_left_right(
            est, lst, adj_mat_mc, ops_durations, self.max_num_jobs, self.max_num_ops, max_num_edges
        )
        action_mask = jax.lax.cond(neighborhood == 5, 
                                   lambda x: get_action_mask_n5(x, self.max_num_ops), 
                                   lambda x: get_action_mask_n6(x, self.max_num_ops, est, num_ops_per_job),
                                   critical_block_info)
        operation_pairs_mask = fully_convert_to_operation_pairs_N5(critical_block_info=critical_block_info, action_mask=action_mask)
        # Time starts at 0
        step_count = jnp.array(0, jnp.int32)

        state = ImprovementState(
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            num_ops_per_job=num_ops_per_job,
            step_count=step_count,
            scheduled_times=scheduled_times,
            adj_mat_pc=adj_mat_pc,
            adj_mat_mc=adj_mat_mc,
            makespan=makespan,
            incumbent_makespan=makespan,
            step_minimum=step_count,
            is_on_critical_path=is_on_critical_path,
            key=key,
            action_mask=action_mask,
            critical_block_info=critical_block_info,
            gap_left_right=gap_left_right,
            est=est,
            lst=lst,
            operation_pairs_mask=operation_pairs_mask,
        )

        return state

    # === Initialization of machine constraint adjacency matrix ===
    # TODO: add other methods, eg. flow due date with most work remaining (FWD-MWR)

    def init_adj_mat_mc(
        self,
        method_id: int,
        plist: chex.Array,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
    ) -> chex.Array:
        """Initialize the machine constraint adjacency matrix according to the chosen method."""

        def method_plist() -> chex.Array:
            return self.init_adj_mat_mc_with_plist(plist, ops_machine_ids, ops_durations)

        def method_spt() -> chex.Array:
            return self.init_adj_mat_mc_with_spt(ops_machine_ids, ops_durations)
        
        def method_fdd_mwr() -> chex.Array:
            return self.init_adj_mat_mc_with_fdd_mwr(ops_machine_ids, ops_durations)

        methods = [
            method_plist,
            method_spt,
            method_fdd_mwr,
        ]

        return jax.lax.switch(method_id, methods)

    def init_adj_mat_mc_with_plist(
        self, plist: chex.Array, ops_machine_ids: chex.Array, ops_durations: chex.Array
    ) -> chex.Array:
        """
        Generate a machine constraint adjacency matrix for an initial job shop schedule,
        based on a job priority list. On each machine, operations are ordered following
        the job priority.

        Args:
            plist: Array (max_num_jobs,) giving the priority order of jobs.
                Only the valid jobs are permuted.
            ops_machine_ids: Array (num_jobs, max_num_ops), -1 for padded operations.
            ops_durations: Array (num_jobs, max_num_ops), -1 for padded operations.

        Returns:
            A (N+2, N+2) adjacency matrix with machine constraints,
            where N = max_num_jobs * max_num_ops.
            Node 0 is the source, node N+1 is the target.
            Edge weights are durations of the preceding operations.
        """

        num_ops_total = self.max_num_jobs * self.max_num_ops
        op_ids = jnp.arange(num_ops_total).reshape((self.max_num_jobs, self.max_num_ops))

        # Reorder all operations, machine IDs, and durations based on the job priority list
        ordered_ops = op_ids[plist].reshape(-1)
        ordered_machines = ops_machine_ids[plist].reshape(-1)
        ordered_durations = ops_durations[plist].reshape(-1)

        # Initialize adjacency matrix (machine constraints only, without source/target)
        adj_mat = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)

        # Separate state: machine_ops (int32), machine_durs (float32)
        init_machine_ops = -jnp.ones((self.num_machines,), dtype=jnp.int32)
        init_machine_durs = -jnp.ones((self.num_machines,), dtype=jnp.float32)

        # Scan over all operations to add machine constraints
        def add_machine_constraint_edge(
            carry: Tuple[chex.Array, chex.Array, chex.Array],
            x: Tuple[int, float, int]
        ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], None]:
            """Add a machine constraint edge to the adjacency matrix."""
            op_id, duration, machine_id = x

            def update_if_valid(
                carry: Tuple[chex.Array, chex.Array, chex.Array]
            ) -> Tuple[chex.Array, chex.Array, chex.Array]:
                machine_ops, machine_durs, adj = carry
                prev_op_id = machine_ops[machine_id]
                prev_duration = machine_durs[machine_id]

                adj = jax.lax.cond(
                    prev_op_id != -1,
                    lambda a: a.at[prev_op_id, op_id].set(prev_duration),
                    lambda a: a,
                    operand=adj,
                )

                machine_ops = machine_ops.at[machine_id].set(op_id)
                machine_durs = machine_durs.at[machine_id].set(duration)
                return machine_ops, machine_durs, adj

            def skip_update(
                carry: Tuple[chex.Array, chex.Array, chex.Array]
            ) -> Tuple[chex.Array, chex.Array, chex.Array]:
                return carry

            new_carry = jax.lax.cond(machine_id != -1, update_if_valid, skip_update, operand=carry)
            return new_carry, None

        (final_machine_ops, final_machine_durs, final_adj_mat), _ = jax.lax.scan(
            add_machine_constraint_edge,
            (init_machine_ops, init_machine_durs, adj_mat),
            (ordered_ops, ordered_durations, ordered_machines),
        )

        # Add source and target nodes (padding with zeros)
        adj_with_source_target = jnp.pad(
            final_adj_mat, ((1, 1), (1, 1)), mode="constant", constant_values=0
        )

        return adj_with_source_target

    def init_adj_mat_mc_with_spt(
        self, ops_machine_ids: chex.Array, ops_durations: chex.Array
    ) -> chex.Array:
        """
        Generate a machine constraint adjacency matrix for an initial job shop schedule,
        based on the rule 'SPT' : shortest processing time.
        The operations are scheduled in the order of increasing processing time.

        Args:
            ops_machine_ids: Array (max_num_jobs, max_num_ops), -1 for padded operations.
            ops_durations: Array (max_num_jobs, max_num_ops), -1 for padded operations.

        Returns:
            A (self.max_jobs * self.max_num_ops + 2, self.max_jobs * self.max_num_ops + 2)
            adjacency matrix with machine constraints.
            Node 0 is the source, node self.max_jobs * self.max_num_ops + 1 is the target.
            Edge weights are durations of the preceding operations.
        """

        num_ops_total = self.max_num_jobs * self.max_num_ops

        # Initialize candidate operations and priorities
        init_cand_ops = jnp.arange(
            0, num_ops_total, self.max_num_ops
        )  # index of operations to be scheduled # shape (max_num_jobs,)

        # More important operations have lower ranks
        init_rank = ops_durations[:, 0]  # shape (max_num_jobs,)
        init_rank = jnp.where(init_rank == -1, jnp.inf, init_rank)
        # Initialize adjacency matrix (machine constraints only, without source/target)
        adj_mat = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)

        # Each machine keeps track of the last scheduled operation and its duration
        # Init with -1 to indicate that no operation has been scheduled yet
        init_machine_ops = -jnp.ones((self.num_machines,), dtype=jnp.int32)
        init_machine_durs = -jnp.ones((self.num_machines,), dtype=jnp.float32)

        def check_remaining_ops(
            carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> jnp.bool_:
            """Check if there are still operations to be scheduled."""
            _, _, _, cand_ops, _, _ = carry
            # Continue while there are still operations to be scheduled
            return ~jnp.all(cand_ops == -1)

        def add_machine_constraint_edge(
            carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
            """Add a machine constraint edge to the adjacency matrix.

            Args:
                carry: Tuple containing (machine_state, adj_mat, cand_ops, rank, ops_durations)

            Returns:
                Tuple containing (updated machine_state,
                                  updated adj_mat,
                                  updated cand_ops,
                                  updated rank,
                                  updated ops_durations)
            """

            machine_ops, machine_durs, adj, cand_ops, rank, ops_durations = carry
            num_ops_per_job = jnp.sum(ops_machine_ids != -1, axis=1)

            # Choose operation to be scheduled next
            job_id = jnp.argmin(rank)
            op_id = cand_ops[job_id]  # Shortest processing time
            machine_id = ops_machine_ids[op_id // self.max_num_ops, op_id % self.max_num_ops]
            duration = ops_durations[op_id // self.max_num_ops, op_id % self.max_num_ops]

            # Get the last operation scheduled on the same machine
            prev_op, prev_duration = machine_ops[machine_id], machine_durs[machine_id]
            cond_prev = prev_op != -1

            # Create edge if there is a previous operation on the same machine
            adj = jax.lax.cond(
                cond_prev,
                lambda a: a.at[prev_op, op_id].set(prev_duration),
                lambda a: a,
                operand=adj,
            )

            # Update the machine state
            machine_ops = machine_ops.at[machine_id].set(op_id)
            machine_durs = machine_durs.at[machine_id].set(duration)

            # Update candidate operations and their ranks
            new_op_id = jnp.where(
                (op_id + 1 < job_id * self.max_num_ops + num_ops_per_job[job_id]), op_id + 1, -1
            )
            cand_ops = cand_ops.at[job_id].set(new_op_id)

            new_rank = jnp.where(
                new_op_id == -1,
                jnp.inf,
                ops_durations[new_op_id // self.max_num_ops, new_op_id % self.max_num_ops],
            )
            rank = rank.at[job_id].set(new_rank)

            return machine_ops, machine_durs, adj, cand_ops, rank, ops_durations

        # Initialize the loop carry state
        init_carry = (init_machine_ops, init_machine_durs, adj_mat, init_cand_ops, init_rank, ops_durations)

        # Run the while loop to build the machine constraint adjacency matrix
        _, _, final_adj_mat, _, _, _ = jax.lax.while_loop(
            check_remaining_ops, add_machine_constraint_edge, init_carry
        )

        # Add source and target nodes (padding with zeros)
        adj_with_source_target = jnp.pad(
            final_adj_mat, ((1, 1), (1, 1)), mode="constant", constant_values=0
        )

        return adj_with_source_target

    def _build_adj_matrix_from_schedule(
        self,
        machine_schedule: chex.Array,
        num_ops_total: int,
    ) -> chex.Array:
        """Builds an adjacency matrix from the final machine schedules."""
        # The correction is here: ensure the matrix is float32.
        adj_matrix = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)

        def update_adj_for_machine(machine_idx: int, adj: chex.Array) -> chex.Array:
            # Create precedence links (op_i -> op_{i+1}) for one machine
            op_ids = machine_schedule[machine_idx]
            senders, receivers = op_ids[:-1], op_ids[1:]

            # Ensure links are only made between valid, scheduled operations
            is_valid_link = (senders != -1) & (receivers != -1)
            
            # Set invalid indices to 0 to prevent out-of-bounds access.
            senders = jnp.where(is_valid_link, senders, 0)
            receivers = jnp.where(is_valid_link, receivers, 0)

            # Update adj[sender, receiver] = 1. JAX will correctly cast the
            # integer `is_valid_link` to float for the float32 matrix.
            return adj.at[senders, receivers].set(is_valid_link.astype(jnp.float32))

        # Loop over all machines to build the full machine-precedence graph
        adj_matrix_mc = jax.lax.fori_loop(0, self.num_machines, update_adj_for_machine, adj_matrix)
        return adj_matrix_mc

    def _run_fdd_mwr_simulation(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        fdd: chex.Array,
    ) -> chex.Array:
        """Runs an iterative scheduling simulation using the FDD/MWR dispatching rule."""
        num_ops_total = self.max_num_jobs * self.max_num_ops

        # --- Initial State ---
        machine_schedule = jnp.full((self.num_machines, num_ops_total), -1, dtype=jnp.int32)
        start_times = jnp.zeros_like(machine_schedule, dtype=jnp.float32)
        end_times = jnp.zeros_like(machine_schedule, dtype=jnp.float32)
        job_predecessor_end_times = jnp.zeros((self.max_num_jobs,), dtype=jnp.float32)

        # The first operation of each job is a candidate
        candidate_ops = jnp.arange(0, num_ops_total, self.max_num_ops, dtype=jnp.int32)
        
        # MWR is the sum of durations for all operations in each job
        dur_masked = jnp.where(ops_durations == -1, 0, ops_durations)
        work_remaining = jnp.sum(dur_masked, axis=1)

        init_carry = (machine_schedule, start_times, end_times, job_predecessor_end_times, candidate_ops, work_remaining)

        def has_candidates(carry_state: tuple) -> bool:
            """Loop continues as long as there are candidate operations."""
            _, _, _, _, candidates, _ = carry_state
            return jnp.any(candidates != -1)

        def schedule_next_op(carry_state: tuple) -> tuple:
            """Schedules one operation based on the FDD/MWR rule."""
            mch_schedule, start_t, end_t, job_pred_ends, cand_ops, work_rem = carry_state

            # --- FDD/MWR Priority Calculation ---
            op_rows, op_cols = cand_ops // self.max_num_ops, cand_ops % self.max_num_ops
            due_dates = fdd[op_rows, op_cols]
            
            # Avoid division by zero if work_remaining is 0
            safe_work_rem = jnp.where(work_rem[op_rows] > 0, work_rem[op_rows], 1e-6)
            ratios = due_dates / safe_work_rem

            # --- Select and Schedule Best Operation ---
            priorities = jnp.where(cand_ops != -1, ratios, jnp.inf)
            best_job_idx = jnp.argmin(priorities)
            op_id = cand_ops[best_job_idx]

            # Schedule the selected operation in the earliest possible slot on its machine
            op_row, op_col = op_id // self.max_num_ops, op_id % self.max_num_ops
            mch_id = ops_machine_ids[op_row, op_col]
            
            updated_mch_sched, updated_starts, updated_ends, job_pred_ends = schedule_op_in_earliest_slot(
                op_id, ops_durations, mch_schedule[mch_id], start_t[mch_id], end_t[mch_id], job_pred_ends, self.max_num_ops
            )

            # --- Update State ---
            mch_schedule = mch_schedule.at[mch_id].set(updated_mch_sched)
            start_t = start_t.at[mch_id].set(updated_starts)
            end_t = end_t.at[mch_id].set(updated_ends)
            
            # Decrease work remaining for the scheduled job
            duration = ops_durations[op_row, op_col]
            work_rem = work_rem.at[best_job_idx].add(-duration)

            # Set the next operation in the job as the new candidate
            num_ops_in_job = jnp.sum(ops_machine_ids[best_job_idx] != -1)
            is_job_finished = (op_col + 1 >= num_ops_in_job)
            new_cand_op = jnp.where(is_job_finished, -1, op_id + 1)
            cand_ops = cand_ops.at[best_job_idx].set(new_cand_op)

            return mch_schedule, start_t, end_t, job_pred_ends, cand_ops, work_rem

        # Run the simulation until all operations are scheduled
        final_carry = jax.lax.while_loop(has_candidates, schedule_next_op, init_carry)
        final_machine_schedule = final_carry[0]
        
        return final_machine_schedule

    def init_adj_mat_mc_with_fdd_mwr(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
    ) -> chex.Array:
        """
        Generates a machine constraint adjacency matrix based on the FDD/MWR rule.
        
        The priority is `FDD / MWR`, where a lower value is higher priority.
        - FDD (Flow Due Date): Cumulative processing time up to the current operation.
        - MWR (Most Work Remaining): Sum of durations of unscheduled operations in the job.
        """
        num_ops_total = self.max_num_jobs * self.max_num_ops
        
        # Pre-calculate FDD for all operations
        dur_masked = jnp.where(ops_durations == -1, 0, ops_durations)
        fdd = jnp.cumsum(dur_masked, axis=1)

        # 1. Simulate the scheduling process to get the final machine operation order
        final_schedule = self._run_fdd_mwr_simulation(
            ops_machine_ids, ops_durations, fdd
        )

        # 2. Build the adjacency matrix from the resulting schedule
        adj_matrix = self._build_adj_matrix_from_schedule(
            final_schedule, num_ops_total
        )
        
        # 3. Pad for source/sink nodes
        return jnp.pad(adj_matrix, ((1, 1), (1, 1)))
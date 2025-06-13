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
    get_critical_operations,
    fully_convert_to_operation_pairs_N5
)
from jumanji.environments.packing.job_shop.improvement.types import ImprovementState
from jumanji.environments.packing.job_shop.types import Scenario


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

    def __call__(self, key: chex.PRNGKey, scenario: Scenario, method_id: int) -> ImprovementState:
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
        critical_block_info = get_critical_operations(
            est, lst, adj_mat_mc, ops_durations, self.max_num_jobs, self.max_num_ops, max_num_edges
        )
        action_mask = get_action_mask_n5(critical_block_info, self.max_num_ops)
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

        # Initialize machine state to keep track of the last scheduled operation and its duration
        # Shape: (num_machines, 2), for each machine contains [last_op_id, duration]
        # Init with -1 to indicate that no operation has been scheduled yet
        init_machine_state = -jnp.ones((self.num_machines, 2), dtype=jnp.int32)

        # Scan over all operations to add machine constraints
        def add_machine_constraint_edge(
            carry: Tuple[chex.Array, chex.Array], x: Tuple[int, int, int]
        ) -> Tuple[Tuple[chex.Array, chex.Array], None]:
            """Add a machine constraint edge to the adjacency matrix.

            Args:
                carry: Tuple containing (machine_state, adj_mat)
                x: Tuple containing (operation id, duration, machine id)

            Returns:
                Tuple containing (updated machine state, updated adjacency matrix)
            """
            op_id, duration, machine_id = x

            def update_if_valid(
                carry: Tuple[chex.Array, chex.Array],
            ) -> Tuple[chex.Array, chex.Array]:
                """Update the adjacency matrix if the operation is a real one (not padded).

                Args:
                    carry: Tuple containing (machine_state, adj_mat)

                Returns:
                    Tuple containing (updated machine state, updated adjacency matrix)
                """
                machine_state, adj = carry
                prev_op_id, prev_duration = machine_state[machine_id]

                adj = jax.lax.cond(
                    prev_op_id != -1,
                    lambda a: a.at[prev_op_id, op_id].set(prev_duration),
                    lambda a: a,
                    operand=adj,
                )

                machine_state = machine_state.at[machine_id].set([op_id, duration])
                return machine_state, adj

            def skip_update(carry: Tuple[chex.Array, chex.Array]) -> Tuple[chex.Array, chex.Array]:
                """Skip the update if the operation is padded."""
                return carry

            new_carry = jax.lax.cond(machine_id != -1, update_if_valid, skip_update, operand=carry)

            return new_carry, None

        (final_machine_state, final_adj_mat), _ = jax.lax.scan(
            add_machine_constraint_edge,
            (init_machine_state, adj_mat),
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
        init_machine_state = -jnp.ones(
            (self.num_machines, 2), dtype=jnp.int32
        )  # shape: (num_machines, [last_op_id, duration])

        def check_remaining_ops(
            carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> jnp.bool_:
            """Check if there are still operations to be scheduled."""
            _, _, cand_ops, _, _ = carry
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

            machine_state, adj, cand_ops, rank, ops_durations = carry
            num_ops_per_job = jnp.sum(ops_machine_ids != -1, axis=1)

            # Choose operation to be scheduled next
            job_id = jnp.argmin(rank)
            op_id = cand_ops[job_id]  # Shortest processing time
            machine_id = ops_machine_ids[op_id // self.max_num_ops, op_id % self.max_num_ops]
            duration = ops_durations[op_id // self.max_num_ops, op_id % self.max_num_ops]

            # Get the last operation scheduled on the same machine
            prev_op, prev_duration = machine_state[machine_id]
            cond_prev = prev_op != -1

            # Create edge if there is a previous operation on the same machine
            adj = jax.lax.cond(
                cond_prev,
                lambda a: a.at[prev_op, op_id].set(prev_duration),
                lambda a: a,
                operand=adj,
            )

            # Update the machine state
            machine_state = machine_state.at[machine_id].set([op_id, duration])

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

            return machine_state, adj, cand_ops, rank, ops_durations

        # Initialize the loop carry state
        init_carry = (init_machine_state, adj_mat, init_cand_ops, init_rank, ops_durations)

        # Run the while loop to build the machine constraint adjacency matrix
        _, final_adj_mat, _, _, _ = jax.lax.while_loop(
            check_remaining_ops, add_machine_constraint_edge, init_carry
        )

        # Add source and target nodes (padding with zeros)
        adj_with_source_target = jnp.pad(
            final_adj_mat, ((1, 1), (1, 1)), mode="constant", constant_values=0
        )

        return adj_with_source_target

    def init_adj_mat_mc_with_fdd_mwr(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
    ) -> chex.Array:
        """
        Generates a machine constraint adjacency matrix based on the FDD/MWR rule:
        Minimum ratio of Flow Due Date to Most Work Remaining. Operations are
        prioritized by selecting the minimum ratio of a job's due date to its
        total remaining processing time.

        Args:
            ops_machine_ids: Array (max_num_jobs, max_num_ops), -1 for padded operations.
            ops_durations: Array (max_num_jobs, max_num_ops), -1 for padded operations.

        Returns:
            A (N+2, N+2) adjacency matrix with machine constraints.
        """
        num_ops_total = self.max_num_jobs * self.max_num_ops
        adj_mat = jnp.zeros((num_ops_total, num_ops_total), dtype=jnp.float32)

        # State for each machine: [last_op_id, duration_of_last_op]
        init_machine_state = -jnp.ones((self.num_machines, 2), dtype=jnp.int32)

        # Initial candidate operations are the first operation of each job
        init_cand_ops = jnp.arange(0, num_ops_total, self.max_num_ops, dtype=jnp.int32)
        
        # Initial due dates are the durations of the first operations of each job
        init_due_dates = ops_durations[:, 0] # can contain -1
        # Initial work remaining is the sum of all operation durations for each job
        init_work_remaining = jnp.sum(jnp.where(ops_durations == -1, 0, ops_durations), axis=1)

        def check_remaining_ops(carry):
            """Continue as long as at least one job has a candidate operation."""
            _, _, cand_ops, _, _ = carry
            return jnp.any(cand_ops != -1)

        def schedule_next_op(carry):
            """Selects and schedules the highest-priority op based on FDD/MWR."""
            machine_state, adj, cand_ops, work_remaining, due_dates = carry

            # Update due dates: add duration of cand_ops if it's not padded
            due_dates = due_dates.at[cand_ops // self.max_num_ops].add(ops_durations[cand_ops // self.max_num_ops, cand_ops % self.max_num_ops])

            # --- Priority Calculation: FDD/MWR Ratio ---
            # Use a small epsilon for stability if work_remaining could be 0
            safe_work_remaining = jnp.where(work_remaining > 0, work_remaining, 1e-6)
            ratios = due_dates / safe_work_remaining
            
            # Set priority to infinity for jobs that are already finished
            priorities = jnp.where(cand_ops != -1, ratios, jnp.inf)
            
            # The best job is the one with the minimum FDD/MWR ratio
            job_id = jnp.argmin(priorities)

            # --- Schedule the selected operation ---
            op_id = cand_ops[job_id]
            op_row, op_col = op_id // self.max_num_ops, op_id % self.max_num_ops
            
            machine_id = ops_machine_ids[op_row, op_col]
            duration = ops_durations[op_row, op_col]

            prev_op, prev_duration = machine_state[machine_id]
            
            # Add edge if there was a previous operation on this machine
            adj = jax.lax.cond(
                prev_op != -1,
                lambda a: a.at[prev_op, op_id].set(prev_duration),
                lambda a: a,
                operand=adj
            )
            machine_state = machine_state.at[machine_id].set(jnp.array([op_id, duration]))

            # --- Update State for the Next Iteration ---
            work_remaining = work_remaining.at[job_id].add(-duration)

            num_ops_in_job = jnp.sum(ops_machine_ids[job_id] != -1)
            is_job_finished = (op_col + 1 >= num_ops_in_job)
            
            new_cand_op = jnp.where(is_job_finished, -1, op_id + 1)
            cand_ops = cand_ops.at[job_id].set(new_cand_op)

            return machine_state, adj, cand_ops, work_remaining, due_dates

        # Run the while loop to build the schedule
        init_carry = (init_machine_state, adj_mat, init_cand_ops, init_work_remaining, init_due_dates)
        _, final_adj_mat, _, _, _ = jax.lax.while_loop(
            check_remaining_ops, schedule_next_op, init_carry
        )
        
        return jnp.pad(final_adj_mat, ((1, 1), (1, 1)), mode="constant")
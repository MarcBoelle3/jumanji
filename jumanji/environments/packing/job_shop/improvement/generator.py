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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.improvement.compute_makespan import (
    compute_est_lst_makespan,
)
from jumanji.environments.packing.job_shop.improvement.get_actions import (
    get_action_mask_n5,
    get_action_mask_n6,
    get_critical_operations_features,
)
from jumanji.environments.packing.job_shop.improvement.scheduling import (
    MethodRegistry,
)
from jumanji.environments.packing.job_shop.improvement.types import (
    BestSolution,
    ImprovementState,
    Neighborhood,
    SchedulingMethod,
)
from jumanji.environments.packing.job_shop.types import Scenario


class ScheduleGenerator(abc.ABC):
    """Abstract base class for generating initial job shop schedules.

    A ScheduleGenerator creates an initial solution for a given scenario by:
    1. Building precedence constraint adjacency matrix (job flow)
    2. Building machine constraint adjacency matrix (machine scheduling)
    3. Computing makespan and critical path information
    4. Generating action masks for improvement neighborhoods
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
            num_jobs: Number of jobs to schedule.
            num_machines: Number of available machines.
            max_num_jobs: Maximum jobs across all instances (static).
            max_num_ops: Maximum operations per job (static).
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_num_jobs = max_num_jobs
        self.max_num_ops = max_num_ops

    @abc.abstractmethod
    def __call__(
        self,
        key: chex.PRNGKey,
        scenario: Scenario,
        method: SchedulingMethod,
        neighborhood: Neighborhood,
    ) -> ImprovementState:
        """Generate initial state for the improvement environment.

        Args:
            key: Random key for stochastic generation.
            scenario: Problem instance containing operations and constraints.
            method: Scheduling method enum (PRIORITY_LIST, SHORTEST_PROCESSING_TIME, etc.).
            neighborhood: Improvement neighborhood (N5 or N6).

        Returns:
            Initial improvement state with schedule and action mask.
        """

    def build_precedence_matrix(
        self, ops_durations: chex.Array, num_ops_per_job: chex.Array
    ) -> chex.Array:
        """Build precedence constraint adjacency matrix.

        Creates job flow constraints where each operation links to the next
        in the same job. Includes source/target nodes for graph completion.

        Args:
            ops_durations: Operation durations (max_num_jobs, max_num_ops).
            num_ops_per_job: Number of operations per job (max_num_jobs,).

        Returns:
            Precedence matrix (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
            with source/target.
        """
        # Build basic job flow structure
        adj_mat = self._build_job_flow_matrix(ops_durations)

        # Add source and target connections
        adj_mat = self._add_source_target_nodes(adj_mat, ops_durations, num_ops_per_job)

        # Apply job masking for valid operations only
        adj_mat = self._apply_job_masking(adj_mat, num_ops_per_job)

        return adj_mat

    def _build_job_flow_matrix(self, ops_durations: chex.Array) -> chex.Array:
        """Create basic job flow adjacency matrix."""
        num_ops_total = self.max_num_jobs * self.max_num_ops
        ops_flat = ops_durations.reshape(-1)
        ops_mask = ops_flat != -1
        durations = jnp.where(ops_mask, ops_flat, 0)

        # Link operation i to operation i+1 with duration weight
        adj_mat = jnp.diag(durations[:-1], k=1)

        # Break links between jobs (last op of job i -> first op of job i+1)
        job_boundaries = jnp.arange(self.max_num_ops - 1, num_ops_total, self.max_num_ops)
        adj_mat = adj_mat.at[job_boundaries, job_boundaries + 1].set(0)

        # Mask invalid operations
        return jnp.where(ops_mask[:, None], adj_mat, 0)

    def _add_source_target_nodes(
        self,
        adj_mat: chex.Array,
        ops_durations: chex.Array,
        num_ops_per_job: chex.Array,
    ) -> chex.Array:
        """Add source (start) and target (end) nodes to adjacency matrix."""
        # Pad matrix for source/target nodes
        adj_mat = jnp.pad(adj_mat, ((1, 1), (1, 1)))

        # Connect source to first operation of each job
        job_starts = 1 + jnp.arange(0, self.max_num_jobs * self.max_num_ops, self.max_num_ops)
        adj_mat = adj_mat.at[0, job_starts].set(1)

        # Connect last operation of each job to target
        safe_num_ops = jnp.maximum(num_ops_per_job, 1)
        job_ends = job_starts + safe_num_ops - 1
        end_durations = ops_durations[jnp.arange(self.max_num_jobs), safe_num_ops - 1]
        adj_mat = adj_mat.at[job_ends, -1].set(jnp.where(num_ops_per_job > 0, end_durations, 0))

        return adj_mat

    def _apply_job_masking(self, adj_mat: chex.Array, num_ops_per_job: chex.Array) -> chex.Array:
        """Mask out connections for non-existent jobs."""
        valid_op_count = self.num_jobs * self.max_num_ops
        num_nodes = adj_mat.shape[0]
        valid_nodes = jnp.arange(num_nodes) <= valid_op_count

        # Preserve target node (last column and row)
        valid_nodes = valid_nodes.at[-1].set(True)
        mask = valid_nodes[:, None] & valid_nodes[None, :]

        # Remove source links to non-existent jobs
        adj_mat = adj_mat.at[0].set(jnp.where(valid_nodes, adj_mat[0], 0))

        return jnp.where(mask, adj_mat, 0)


class RandomScheduleGenerator(ScheduleGenerator):
    """Generate initial schedules using configurable scheduling methods.

    Supports multiple scheduling algorithms through the MethodRegistry:
    - 'priority_list': Random job priority ordering
    - 'shortest_processing_time': SPT dispatching rule
    - 'flow_due_date_most_work': FDD/MWR ratio scheduling
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_jobs: int,
        max_num_ops: int,
    ):
        """Initialize the generator with automatic scheduling method discovery.

        Args:
            num_jobs: Number of jobs in the scenario.
            num_machines: Number of machines.
            max_num_jobs: Maximum number of jobs.
            max_num_ops: Maximum number of operations per job.
        """
        super().__init__(num_jobs, num_machines, max_num_jobs, max_num_ops)
        # Automatically get all registered scheduling methods with problem parameters
        self.scheduling_methods = MethodRegistry.get_instances(
            num_jobs, num_machines, max_num_jobs, max_num_ops
        )

    def __call__(
        self,
        key: chex.PRNGKey,
        scenario: Scenario,
        method: SchedulingMethod,
        neighborhood: Neighborhood,
    ) -> ImprovementState:
        """Function to generate the initial state.

        Args:
            key: Random key for stochastic generation.
            scenario: Problem instance containing operations and constraints.
            method: Scheduling method enum.
            neighborhood: Improvement neighborhood (N5 or N6).

        Returns:
            Initial improvement state with schedule and action mask.
        """
        max_num_edges = (1 + 2 * self.max_num_ops) * self.max_num_jobs

        # Extract scenario data
        ops_machine_ids = scenario.ops_machine_ids
        ops_durations = scenario.ops_durations
        num_ops_per_job = scenario.num_ops_per_job

        # Build adjacency matrices
        adj_mat_pc = self.build_precedence_matrix(ops_durations, num_ops_per_job)
        adj_mat_mc = self._build_machine_constraints(method, key, ops_machine_ids, ops_durations)

        # Compute schedule metrics
        adj_mat = jnp.maximum(adj_mat_pc, adj_mat_mc)
        est, lst, makespan = compute_est_lst_makespan(adj_mat, ops_durations, max_num_edges)

        scheduled_times = est[1:-1].reshape((self.max_num_jobs, self.max_num_ops))
        is_on_critical_path = (est[1:-1] == lst[1:-1]).reshape(
            (self.max_num_jobs, self.max_num_ops)
        )

        # Generate action mask for improvement
        critical_block_info, gap_left_right = get_critical_operations_features(
            est, lst, adj_mat_mc, ops_durations, self.max_num_jobs, self.max_num_ops, max_num_edges
        )

        action_mask = jax.lax.cond(
            neighborhood == Neighborhood.N5,
            lambda x: get_action_mask_n5(x, self.max_num_ops),
            lambda x: get_action_mask_n6(x, self.max_num_ops, est, num_ops_per_job),
            critical_block_info,
        )

        # Initialize state
        step_count = jnp.array(0, jnp.int32)
        best_solution = BestSolution(
            scheduled_times=scheduled_times,
            adj_mat_pc=adj_mat_pc,
            adj_mat_mc=adj_mat_mc,
            is_on_critical_path=is_on_critical_path,
            action_mask=action_mask,
            critical_block_info=critical_block_info,
            gap_left_right=gap_left_right,
            est=est,
            lst=lst,
        )

        return ImprovementState(
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
            best_solution_so_far=best_solution,
            step_since_best=jnp.array(0, jnp.int32),
        )

    def _build_machine_constraints(
        self,
        method: SchedulingMethod,
        key: chex.PRNGKey,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
    ) -> chex.Array:
        """
        Args:
            method: Scheduling method enum.
            key: Random key for stochastic methods.
            ops_machine_ids: Machine assignments (max_num_jobs, max_num_ops).
            ops_durations: Operation durations (max_num_jobs, max_num_ops).

        Returns:
            Machine constraint adjacency matrix
            (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2).
        """

        branches = [
            lambda k, method=method_instance: method.build_machine_adjacency_matrix(
                ops_machine_ids,
                ops_durations,
                key=k,
            )
            for method_instance in self.scheduling_methods
        ]

        return jax.lax.switch(method, branches, key)

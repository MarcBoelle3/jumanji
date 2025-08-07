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

from functools import cached_property
from typing import Optional, Sequence, Tuple, Literal

import chex
import jax
import jax.numpy as jnp
import matplotlib

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.packing.job_shop.improvement.compute_makespan import (
    compute_est_lst_makespan,
)
from jumanji.environments.packing.job_shop.improvement.generator import (
    RandomScheduleGenerator,
    ScheduleGenerator,
)
from jumanji.environments.packing.job_shop.improvement.get_actions import (
    get_action_mask_n5,
    get_critical_operations,
    get_critical_operations_plus_empty_space_left_right,
    select_operations_to_switch,
    fully_convert_to_operation_pairs_N5,
    get_action_mask_n6,
    CBFields
)

from jumanji.environments.packing.job_shop.improvement.types import ImprovementState, Observation, BestSolution
from jumanji.environments.packing.job_shop.improvement.update_sol import update_disjunctive_graph
from jumanji.environments.packing.job_shop.scenario_generator import (
    RandomScenarioGenerator,
    ScenarioGenerator,
)
from jumanji.environments.packing.job_shop.viewer import JobShopViewer
from jumanji.types import TimeStep, restart, termination, transition, truncation
from jumanji.viewer import Viewer


class JobShop(Environment[ImprovementState, specs.MultiDiscreteArray, Observation]):
    """Job Shop Scheduling Problem (JSSP) environment for improvement heuristics.

    This environment implements a JSSP where the goal is to improve an initial solution
    through local search operations. The environment supports different initialization
    methods and reward types.
    """

    def __init__(
        self,
        scenario_generator: Optional[ScenarioGenerator] = None,
        schedule_generator: Optional[ScheduleGenerator] = None,
        viewer: Optional[Viewer[ImprovementState]] = None,
        time_limit: int = 500,
        neighborhood: int = 5,
        reward_type: Literal["incumbent", "composed"] = "incumbent",
        reward_scale: float = 0.3,
        mask_last_action: bool = False,
        restart_from_best: bool = False,
        nb_steps_before_restart: Optional[int] = 50
    ):
        """Initialize the Job Shop Improvement environment.

        Args:
            scenario_generator: Generator object for creating problem instances.
            schedule_generator: Generator object for creating initial schedules.
            viewer: Viewer object for rendering.
            time_limit: Maximum number of steps per episode.
            neighborhood: Neighborhood type (5 or 6).
            reward_type: Type of reward calculation ("incumbent" or "composed").
            reward_scale: Scale factor for composed reward.
            mask_last_action: Whether to mask the last action to prevent immediate reversal.
            restart_from_best: Whether to restart from the best solution when stuck.
            nb_steps_before_restart: Number of steps without improvement before restarting.
        """

        self.scenario_generator = scenario_generator or RandomScenarioGenerator(
            max_num_jobs=20,
            max_num_ops=10,
            max_op_duration=6,
        )
        self.schedule_generator = schedule_generator or RandomScheduleGenerator(
            num_jobs=20, num_machines=10, max_num_ops=10, max_num_jobs=20
        )

        # Initialize static parameters
        self.max_num_jobs = self.scenario_generator.max_num_jobs
        self.max_op_duration = self.scenario_generator.max_op_duration
        self.max_num_ops = self.scenario_generator.max_num_ops
        self.max_num_edges_mc = self.max_num_ops * self.max_num_jobs #upper bound
        self.max_num_edges_pc = self.max_num_jobs * (self.max_num_ops +1)
        self.max_num_edges = self.max_num_edges_mc + self.max_num_edges_pc
        self.neighborhood = neighborhood
        # Initialize dynamic parameters
        self.num_jobs = self.schedule_generator.num_jobs
        self.num_machines = self.schedule_generator.num_machines
        self.time_limit = time_limit

        # Initialize reward parameters
        self.reward_type = reward_type
        self.reward_scale = reward_scale

        # Initialize mask last action
        self.mask_last_action = mask_last_action

        # Initialize restart from best parameters
        self.restart_from_best = restart_from_best
        self.nb_steps_before_restart = nb_steps_before_restart

        super().__init__()

        # Create viewer used for rendering
        self._viewer = viewer or JobShopViewer(
            "JobShop",
            self.num_jobs,
            self.num_machines,
            self.max_num_ops,
            self.max_op_duration,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "JobShop environment:",
                f" - scenario_generator: {self.scenario_generator}",
                f" - schedule_generator: {self.schedule_generator}",
                f" - num_jobs: {self.num_jobs}",
                f" - num_machines: {self.num_machines}",
                f" - max_num_ops: {self.max_num_ops}",
                f" - max_op_duration: {self.max_op_duration}",
            ]
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the `JobShop` environment.

        Returns:
            Spec containing the specifications for all the `Observation` fields:
            - ops_machine_ids: BoundedArray (int32) of shape (num_jobs, max_num_ops).
            - ops_durations: BoundedArray (float32) of shape (num_jobs, max_num_ops).
            - adj_mat_pc: BoundedArray (int32) of shape (num_jobs, num_jobs).
            - adj_mat_mc: BoundedArray (int32) of shape (num_jobs, num_jobs).
            - makespan: BoundedArray (int32) of shape ().
        """
        ops_machine_ids = specs.BoundedArray(
            shape=(self.max_num_jobs, self.max_num_ops),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.num_machines - 1,
            name="ops_machine_ids",
        )
        ops_durations = specs.BoundedArray(
            shape=(self.max_num_jobs, self.max_num_ops),
            dtype=jnp.float32,
            minimum=-1,
            maximum=self.max_op_duration,
            name="ops_durations",
        )
        edges_pc = specs.Array(
            shape=(self.max_num_edges_pc, 2),
            dtype=jnp.int32,
            name="edges_pc",
        )
        edges_mc = specs.Array(
            shape=(self.max_num_edges_mc, 2),
            dtype=jnp.int32,
            name="edges_mc",
        )
        makespan = specs.Array(
            shape=(),
            dtype=jnp.float32,
            name="makespan",
        )
        incumbent_makespan = specs.Array(
            shape=(),
            dtype=jnp.float32,
            name="incumbent_makespan",
        )
        action_mask = specs.Array(
            shape=(self.max_num_ops * self.max_num_jobs, 2),
            dtype=bool,
            name="action_mask",
        )
        observation_features = specs.Array(
            shape=(self.max_num_jobs * self.max_num_ops + 2, 5),
            dtype=jnp.float32,
            name="observation_features",
        )
        operation_pairs_mask = specs.Array(
            shape=(self.max_num_ops * self.max_num_jobs, self.max_num_ops * self.max_num_jobs),
            dtype=jnp.bool_,
            name="operation_pairs_mask",
        )
        num_machines = specs.Array(
            shape=(),
            dtype=jnp.int32,
            name="num_machines",
        )
        extra_features = specs.Array(
            shape=(self.max_num_ops * self.max_num_jobs, 3),
            dtype=jnp.int32,
            name="extra_features",
        )
        return specs.Spec(
            constructor=Observation,
            name="ObservationSpec",
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            edges_pc=edges_pc,
            edges_mc=edges_mc,
            makespan=makespan,
            incumbent_makespan=incumbent_makespan,
            action_mask=action_mask,
            observation_features=observation_features,
            operation_pairs_mask=operation_pairs_mask,
            num_machines=num_machines,
            extra_features=extra_features,
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Specifications of the action in the `JobShopImprovement` environment.
        The action gives a tuple (action_idx, left_or_right), where action_idx is the index of the operation to be switched
        and left_or_right is a direction of the move.  For left_or_right:
        - 0: start stays at its position and end moves before it.
        - 1: end stays at its position and start moves after it.
        The direction is dummy for N5 neighborhood, as it gives the same result after masking,
        but required for N6 neighborhood as it distinguishes between the two possible moves.

        Returns:
            action_spec: a `specs.Array` spec.
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.max_num_ops * self.max_num_jobs, 2], dtype=jnp.int32),
            name="action",
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[ImprovementState, TimeStep[Observation]]:
        """Resets the environment by creating a new problem instance and initialising the state
        and timestep.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: the environment state after the reset.
            timestep: the first timestep returned by the environment after the reset.
        """
        # Generate a new problem instance
        scenario = self.scenario_generator(key, self.num_jobs, self.num_machines)
        state = self.schedule_generator(
            scenario.key, scenario, method_id=2, neighborhood=self.neighborhood
        )  # for now, method is fdd/mwr

        obs = self._observation_from_state(state)
        timestep = restart(observation=obs)

        return state, timestep

    def step(
        self, state: ImprovementState, action: chex.Array
    ) -> Tuple[ImprovementState, TimeStep[Observation]]:
        """Updates the environment state by applying the given action and computing the new makespan.

        The function:
        1. Updates the graph topology by combining precedence constraints and machine constraints
        2. Computes the makespan using forward and backward passes on the adjacency matrix
        3. Calculates the reward
        4. Updates the incumbent and current objectives
        5. Increments the iteration counter
        6. Gets new feasible actions
        7. Creates a new state and timestep

        Args:
            state: the environment state containing the current job shop configuration.
            action: the action to take, representing a pair of operations to be switched.

        Returns:
            state: the updated environment state with the new configuration.
            timestep: the updated timestep containing the new observation and reward.
        """

        # Convert action to start, end, move_start_end indices
        action_ops_pair = select_operations_to_switch(state.critical_block_info, action, neighborhood=self.neighborhood)

        # Update graph topology based on action
        new_adj_mat_mc = update_disjunctive_graph(
            state.adj_mat_mc, state.ops_durations, action_ops_pair
        )

        adj_mat = jnp.maximum(
            state.adj_mat_pc, new_adj_mat_mc
        )  # to handle the case where job and machine constraints are in conflict
        # Compute makespan using forward and backward pass
        est, lst, makespan = compute_est_lst_makespan(
            adj_mat, state.ops_durations, self.max_num_edges
        )
        new_scheduled_times = est[1:-1].reshape(
            (self.max_num_jobs, self.max_num_ops)
        )  # discard source and target and set scheduled times to earliest start times
        new_scheduled_times = jnp.where(
            new_scheduled_times == -jnp.inf, -1, new_scheduled_times
        )  # replace -inf with -1
        # Identify critical operations
        is_on_critical_path = (est[1:-1] == lst[1:-1]).reshape(
            (self.max_num_jobs, self.max_num_ops)
        )

        # Compute reward
        if self.reward_type == "incumbent":
            reward = jnp.maximum(state.incumbent_makespan - makespan, 0)
        elif self.reward_type == "composed":
            reward = jnp.maximum(state.incumbent_makespan - makespan, 0) + self.reward_scale * (state.makespan - makespan)
        incumbent_makespan = jnp.minimum(state.incumbent_makespan, makespan)
        step_minimum = jnp.where(
            makespan < state.incumbent_makespan,
            state.step_count + 1,
            state.step_minimum,
        )
        # reward = state.makespan - makespan

        action_mask, critical_block_info, gap_left_right = self._create_action_mask(
            est, lst, new_adj_mat_mc, state.ops_durations, state.num_ops_per_job
        )
        
        # Mask the last action to prevent immediate reversal
        if self.mask_last_action:
            action_mask = self._mask_last_action(action_mask, action, action_ops_pair)
        
        # Check if there are any valid actions in the action mask
        has_valid_actions = jnp.any(action_mask)
        #jax.debug.print("has_valid_actions: {}", has_valid_actions)
        #jax.debug.breakpoint()
        #New! for masking operation pairs:
        operation_pairs_mask = fully_convert_to_operation_pairs_N5(critical_block_info, action_mask)

        # Create new state
        new_state = ImprovementState(
            ops_machine_ids=state.ops_machine_ids,
            ops_durations=state.ops_durations,
            num_ops_per_job=state.num_ops_per_job,
            step_count=state.step_count + 1,
            scheduled_times=new_scheduled_times,
            adj_mat_pc=state.adj_mat_pc,
            adj_mat_mc=new_adj_mat_mc,
            makespan=makespan,
            incumbent_makespan=incumbent_makespan,
            step_minimum=step_minimum,
            is_on_critical_path=is_on_critical_path,
            action_mask=action_mask,
            operation_pairs_mask=operation_pairs_mask,
            critical_block_info=critical_block_info,
            gap_left_right=gap_left_right,
            est=est,
            lst=lst,
            key=state.key,
            best_solution_so_far=state.best_solution_so_far,
            step_since_best=state.step_since_best,
        )

        # Update restart from best logic
        improved = makespan < state.incumbent_makespan
        new_state = self._update_restart_from_best(state, new_state, improved)

        # Create observation
        next_obs = self._observation_from_state(new_state)

        done = state.step_count >= self.time_limit

        branches = [
            truncation,
            termination,
            transition,
        ]

        index = jnp.select(
            [done, ~has_valid_actions],
            [0, 1],
            default=2
        )

        timestep = jax.lax.switch(
            index,
            branches,
            reward,
            next_obs,
        )

        return new_state, timestep

    def animate(
        self,
        states: Sequence[ImprovementState],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the Jobshop environment based on the sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(states, interval, save_path)

    def _create_action_mask(
        self, est: chex.Array, lst: chex.Array, adj_mat_mc: chex.Array, ops_durations: chex.Array, num_ops_per_job: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Create the action mask corresponding to N5 neighborhood."""

        critical_block_info, gap_left_right = get_critical_operations_plus_empty_space_left_right(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.max_num_jobs,
            self.max_num_ops,
            self.max_num_edges,
        )
        action_mask = jax.lax.cond(
            self.neighborhood == 5,
            lambda x: get_action_mask_n5(x, self.max_num_ops),
            lambda x: get_action_mask_n6(x, self.max_num_ops, est, num_ops_per_job),
            critical_block_info
        )
        return action_mask, critical_block_info, gap_left_right

    def _mask_last_action(self, action_mask: chex.Array, last_action: chex.Array, action_ops_pair: chex.Array) -> chex.Array:
        """Mask the last action to prevent immediate reversal.
        
        Args:
            action_mask: Current action mask of shape (max_num_ops * max_num_jobs, 2)
            last_action: Last action taken, array of shape (2,) with [action_idx, direction]
            action_ops_pair: Array of shape (3,) with [start_op_idx, end_op_idx, _]
        Returns:
            Updated action mask with last action masked out
        """
        action_idx, direction = last_action[0], last_action[1]
        mask_update = jnp.ones_like(action_mask, dtype=bool)

        if self.neighborhood == 6:
            # Prevent immediate reversal: if last action was direction 0, mask direction 1 and vice versa
            opposite_direction = 1 - direction
            # Create a mask that sets the last action to False
            mask_update = mask_update.at[action_idx, opposite_direction].set(False)
        
        elif self.neighborhood == 5:
            action_start, action_end, _ = action_ops_pair #action_start always before action_end
            idx_to_update = (1 - direction) * action_start + direction * action_end 
            #idx_to_update : action_start if direction is 0(left), action_end if direction is 1(right)
            mask_update = mask_update.at[idx_to_update, direction].set(False)
        
        # Apply the mask
        return action_mask & mask_update

    def _create_best_solution(self, state: ImprovementState) -> BestSolution:
        """Create a BestSolution object from the current state."""
        return BestSolution(
            scheduled_times=state.scheduled_times,
            adj_mat_pc=state.adj_mat_pc,
            adj_mat_mc=state.adj_mat_mc,
            is_on_critical_path=state.is_on_critical_path,
            action_mask=state.action_mask,
            operation_pairs_mask=state.operation_pairs_mask,
            critical_block_info=state.critical_block_info,
            gap_left_right=state.gap_left_right,
            est=state.est,
            lst=state.lst,
        )

    def _restart_from_best_solution(self, state: ImprovementState) -> ImprovementState:
        """Restart the state from the best solution found so far."""
        best_sol = state.best_solution_so_far
        return state.replace(
            scheduled_times=best_sol.scheduled_times,
            adj_mat_pc=best_sol.adj_mat_pc,
            adj_mat_mc=best_sol.adj_mat_mc,
            is_on_critical_path=best_sol.is_on_critical_path,
            action_mask=best_sol.action_mask,
            operation_pairs_mask=best_sol.operation_pairs_mask,
            critical_block_info=best_sol.critical_block_info,
            gap_left_right=best_sol.gap_left_right,
            est=best_sol.est,
            lst=best_sol.lst,
            makespan=state.incumbent_makespan,
            step_since_best=0,
        )

    def _update_restart_from_best(self, state: ImprovementState, new_state: ImprovementState, improved: bool) -> ImprovementState:
        """Update the restart from best logic using JAX-compatible operations."""
        
        def no_restart_update():
            """Return new_state unchanged when restart is disabled."""
            return new_state
        
        def restart_enabled_update():
            """Handle restart logic when enabled."""
            
            def improvement_update():
                """Update best solution and reset counter when improved."""
                best_solution = self._create_best_solution(new_state)
                return new_state.replace(
                    best_solution_so_far=best_solution,
                    step_since_best=0
                )
            
            def no_improvement_update():
                """Handle case when no improvement occurred."""
                new_step_since_best = state.step_since_best + 1
                
                def restart_needed():
                    """Restart from best solution when threshold reached."""
                    return self._restart_from_best_solution(new_state)
                
                def continue_counting():
                    """Continue counting steps since best."""
                    return new_state.replace(step_since_best=new_step_since_best)
                
                # Check if restart is needed
                return jax.lax.cond(
                    new_step_since_best >= self.nb_steps_before_restart,
                    restart_needed,
                    continue_counting
                )
            
            # Choose between improvement and no improvement cases
            return jax.lax.cond(
                improved,
                improvement_update,
                no_improvement_update
            )
        
        # Main conditional: check if restart is enabled
        return jax.lax.cond(
            self.restart_from_best,
            restart_enabled_update,
            no_restart_update
        )

    def _observation_from_state(self, state: ImprovementState) -> Observation:
        """Converts a job shop environment state to an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            observation: `Observation` object containing the observation of the environment.
        """

        adj_mat_mc = state.adj_mat_mc
        adj_mat_pc = state.adj_mat_pc

        senders_mc, receivers_mc = jnp.nonzero(adj_mat_mc > 0, size=self.max_num_edges_mc, fill_value=-1)
        senders_pc, receivers_pc = jnp.nonzero(adj_mat_pc > 0, size=self.max_num_edges_pc, fill_value=-1)
        edges_mc = jnp.stack([senders_mc, receivers_mc], axis=-1)  # shape (num_edges, 2)
        edges_pc = jnp.stack([senders_pc, receivers_pc], axis=-1)  # shape (num_edges, 2)

        #add original paper observation features: ops_duration, est and lst for each operation

        empty_space_left = state.gap_left_right[:, 0]
        empty_space_right = state.gap_left_right[:, 1]
        empty_space_left_plus_source_target = jnp.concatenate([jnp.array([0.0]), empty_space_left, jnp.array([0.0])])
        empty_space_right_plus_source_target = jnp.concatenate([jnp.array([0.0]), empty_space_right, jnp.array([0.0])])
        observation_features = jnp.stack([
            jnp.pad(state.ops_durations.reshape(-1), (1, 1), mode='constant', constant_values=0),
            state.est,
            state.lst,
            empty_space_left_plus_source_target,
            empty_space_right_plus_source_target
        ], axis=-1) #shape (max_num_ops*max_num_jobs + 2, 5)

        #Where ops_durations is -1, mask to zero
        observation_features = jnp.where(observation_features[:, 0].reshape(-1, 1) == -1, jnp.array([-1, 0, 0, 0, 0]), observation_features)
        
        extra_features = jnp.stack([
            state.critical_block_info[:, CBFields.BLOCK_ID],
            state.critical_block_info[:, CBFields.IS_LEFT],
            state.critical_block_info[:, CBFields.IS_RIGHT]
        ], axis=-1)
        
        return Observation(
            ops_machine_ids=state.ops_machine_ids,
            ops_durations=state.ops_durations,
            edges_pc=edges_pc,
            edges_mc=edges_mc,
            makespan=state.makespan,
            incumbent_makespan=state.incumbent_makespan,
            action_mask=state.action_mask,
            operation_pairs_mask=state.operation_pairs_mask,
            observation_features=observation_features,
            num_machines=self.num_machines,
            extra_features=extra_features,
        )

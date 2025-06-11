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
from typing import Optional, Sequence, Tuple

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
    select_operations_to_switch,
    fully_convert_to_operation_pairs_N5
)
from jumanji.environments.packing.job_shop.improvement.types import ImprovementState, Observation
from jumanji.environments.packing.job_shop.improvement.update_sol import update_disjunctive_graph
from jumanji.environments.packing.job_shop.scenario_generator import (
    RandomScenarioGenerator,
    ScenarioGenerator,
)
from jumanji.environments.packing.job_shop.viewer import JobShopViewer
from jumanji.types import TimeStep, restart, termination, transition
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
    ):
        """Initialize the Job Shop Improvement environment.

        Args:
            generator: Generator object for creating problem instances.
            fea_norm_const: Normalization constant for features.
            time_limit: Maximum number of steps per episode.
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
        self.max_num_edges = (1 + 2 * self.max_num_ops) * self.max_num_jobs

        # Initialize dynamic parameters
        self.num_jobs = self.schedule_generator.num_jobs
        self.num_machines = self.schedule_generator.num_machines
        self.time_limit = time_limit

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
            - ops_durations: BoundedArray (int32) of shape (num_jobs, max_num_ops).
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
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.max_op_duration,
            name="ops_durations",
        )
        edges_pc = specs.Array(
            shape=(self.max_num_edges, 2),
            dtype=jnp.int32,
            name="edges_pc",
        )
        edges_mc = specs.Array(
            shape=(self.max_num_edges, 2),
            dtype=jnp.int32,
            name="edges_mc",
        )
        makespan = specs.Array(
            shape=(),
            dtype=jnp.float32,
            name="makespan",
        )
        action_mask = specs.Array(
            shape=(self.max_num_ops * self.max_num_jobs, 2),
            dtype=bool,
            name="action_mask",
        )
        observation_features = specs.Array(
            shape=(self.max_num_jobs * self.max_num_ops, 3),
            dtype=jnp.int32,
            name="observation_features",
        )
        operation_pairs_mask = specs.Array(
            shape=(self.max_num_ops * self.max_num_jobs, self.max_num_ops * self.max_num_jobs),
            dtype=jnp.bool_,
            name="operation_pairs_mask",
        )
        return specs.Spec(
            constructor=Observation,
            name="ObservationSpec",
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            edges_pc=edges_pc,
            edges_mc=edges_mc,
            makespan=makespan,
            action_mask=action_mask,
            observation_features=observation_features,
            operation_pairs_mask=operation_pairs_mask,
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
            scenario.key, scenario, method_id=2
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
        # Neighborhood 5 for the moment
        action_ops_pair = select_operations_to_switch(state.critical_block_info, action, 5)
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
        reward = state.makespan - makespan

        action_mask, critical_block_info = self._create_action_mask(
            est, lst, new_adj_mat_mc, state.ops_durations
        )
        # Check if there are any valid actions in the action mask
        has_valid_actions = jnp.any(action_mask)

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
            is_on_critical_path=is_on_critical_path,
            action_mask=action_mask,
            operation_pairs_mask=operation_pairs_mask,
            critical_block_info=critical_block_info,
            est=est,
            lst=lst,
            key=state.key,
        )

        # Create observation
        next_obs = self._observation_from_state(new_state)

        done = state.step_count >= self.time_limit

        timestep = jax.lax.cond(
            done | ~has_valid_actions,
            termination,
            transition,
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
        self, est: chex.Array, lst: chex.Array, adj_mat_mc: chex.Array, ops_durations: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Create the action mask corresponding to N5 neighborhood."""

        critical_block_info = get_critical_operations(
            est,
            lst,
            adj_mat_mc,
            ops_durations,
            self.max_num_jobs,
            self.max_num_ops,
            self.max_num_edges,
        )
        action_mask = get_action_mask_n5(critical_block_info, self.max_num_ops)
        return action_mask, critical_block_info

    def _observation_from_state(self, state: ImprovementState) -> Observation:
        """Converts a job shop environment state to an observation.

        Args:
            state: `State` object containing the dynamics of the environment.

        Returns:
            observation: `Observation` object containing the observation of the environment.
        """

        adj_mat_mc = state.adj_mat_mc
        adj_mat_pc = state.adj_mat_pc

        senders_mc, receivers_mc = jnp.nonzero(adj_mat_mc > 0, size=self.max_num_edges, fill_value=-1)
        senders_pc, receivers_pc = jnp.nonzero(adj_mat_pc > 0, size=self.max_num_edges, fill_value=-1)
        edges_mc = jnp.stack([senders_mc, receivers_mc], axis=-1)  # shape (num_edges, 2)
        edges_pc = jnp.stack([senders_pc, receivers_pc], axis=-1)  # shape (num_edges, 2)

        #add original paper observation features: ops_duration, est and lst for each operation
        observation_features = jnp.stack([
            state.ops_durations.reshape(-1),
            state.est[1:-1],
            state.lst[1:-1],
        ], axis=-1) #shape (max_num_ops*max_num_jobs, 3)

        #Where ops_durations is -1, mask to zero
        observation_features = jnp.where(state.ops_durations.reshape(-1, 1) == -1, jnp.array([-1, 0, 0]), observation_features)

        #Normalize observation features: divide duration by 99, est and lst by 1000, as in the paper

        observation_features = observation_features.at[:, 0].divide(99.0)
        observation_features = observation_features.at[:, 1:].divide(1000.0)

        return Observation(
            ops_machine_ids=state.ops_machine_ids,
            ops_durations=state.ops_durations,
            edges_pc=edges_pc,
            edges_mc=edges_mc,
            makespan=state.makespan,
            action_mask=state.action_mask,
            operation_pairs_mask=state.operation_pairs_mask,
            observation_features=observation_features,
        )

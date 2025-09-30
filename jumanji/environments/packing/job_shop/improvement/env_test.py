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
import pytest

from jumanji.environments.packing.job_shop.improvement.env import JobShop
from jumanji.environments.packing.job_shop.improvement.types import (
    ImprovementState,
    Neighborhood,
    Observation,
    SchedulingMethod,
)
from jumanji.environments.packing.job_shop.improvement.update_sol import (
    update_disjunctive_graph,
)
from jumanji.environments.packing.job_shop.scenario_generator import (
    RandomScenarioGenerator,
)
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
    make_random_select_action_fn,
)
from jumanji.types import TimeStep


class TestJobShop:
    """Original test class for JobShop environment."""

    def test_job_shop__reset(self, job_shop_env: JobShop) -> None:
        """Test that the environment is reset correctly."""
        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        assert jnp.all(
            state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(
            state.scheduled_times
            == jnp.array(
                [
                    [0, 3, 6],
                    [3, 5, 9],
                    [5, 9, -1],
                ],
                jnp.int32,
            )
        )
        assert jnp.all(state.num_ops_per_job == jnp.array([3, 3, 2], jnp.int32))
        assert state.step_count == jnp.array(0, jnp.int32)

    def test_job_shop__reset_jit(self, job_shop_env: JobShop) -> None:
        """Confirm that the reset is only compiled once when jitted."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(job_shop_env.reset, n=1))
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        # Call again to check it does not compile twice
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, ImprovementState)

    def test_job_shop_adj_mat_pc(self, job_shop_env: JobShop) -> None:
        """Test adjacency matrix for precedence constraints."""
        key = jax.random.PRNGKey(0)
        state, _ = job_shop_env.reset(key)
        jax.debug.print("state.adj_mat_pc: {}", (state.adj_mat_pc,))
        assert jnp.all(
            state.adj_mat_pc
            == jnp.array(
                [
                    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

    def test_job_shop_update_disjunctive_graph(self, job_shop_env: JobShop) -> None:
        """Test disjunctive graph update functionality."""
        key = jax.random.PRNGKey(0)
        state, _ = job_shop_env.reset(key)

        assert jnp.all(
            state.adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

        # Test that the update is the same for both actions
        dummy_ops_pair1 = jnp.array(
            [6, 5, 0], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 5 coming left to 6
        dummy_ops_pair2 = jnp.array(
            [6, 5, 1], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 6 coming right to 5
        new_adj_mat_mc = update_disjunctive_graph(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair1
        )
        new_adj_mat_mc_2 = update_disjunctive_graph(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair2
        )
        assert jnp.all(
            new_adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

        assert jnp.all(new_adj_mat_mc_2 == new_adj_mat_mc)

    def test_job_shop_update_disjunctive_graph_jitted(self, job_shop_env: JobShop) -> None:
        """Test disjunctive graph update under JIT compilation."""
        key = jax.random.PRNGKey(0)
        state, _ = job_shop_env.reset(key)

        dummy_ops_pair1 = jnp.array(
            [6, 5, 0], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 5 coming left to 6
        dummy_ops_pair2 = jnp.array(
            [6, 5, 1], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 6 coming right to 5
        new_adj_mat_mc = jax.jit(update_disjunctive_graph)(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair1
        )
        new_adj_mat_mc_2 = jax.jit(update_disjunctive_graph)(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair2
        )
        assert jnp.all(
            new_adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )
        assert jnp.all(new_adj_mat_mc_2 == new_adj_mat_mc)

    def test_job_shop__step(self, job_shop_env: JobShop) -> None:
        """Test 2 steps of the dummy instance."""

        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        # STEP T=0 -> T=1
        # Action: switch operation 6 to the right
        action = jnp.array([6, 1], jnp.int32)
        next_state, next_timestep = job_shop_env.step(state, action)

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )

        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [0, 3, 6],
                    [3, 5, 6],
                    [10, 14, -1],
                ],
                jnp.int32,
            )
        )
        assert next_state.step_count == 1
        assert next_state.makespan == 17

    def test_job_shop__step_jit(self, job_shop_env: JobShop) -> None:
        """Confirm that the step is only compiled once when jitted."""
        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        # Action : switch operation 6 to the right
        action = jnp.array([6, 1], jnp.int32)
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(job_shop_env.step, n=1))
        _, _ = step_fn(state, action)

        # Call again to check it does not compile twice
        next_state, next_timestep = step_fn(state, action)
        assert isinstance(next_timestep, TimeStep)
        assert isinstance(next_state, ImprovementState)

    def test_job_shop_env__does_not_smoke(self, job_shop_env: JobShop) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(
            job_shop_env,
            select_action=make_random_select_action_fn(job_shop_env.action_spec),
        )

    def test_job_shop_env__specs_does_not_smoke(self, job_shop_env: JobShop) -> None:
        """Test that we can access specs without any errors."""
        check_env_specs_does_not_smoke(job_shop_env)


class TestJobShopStepComprehensive:
    """Comprehensive test suite for JobShop step functionality."""

    @pytest.fixture
    def step_job_shop_env(self) -> JobShop:
        """Create a JobShop environment for step testing."""
        from jumanji.environments.packing.job_shop.improvement.generator import ScheduleGenerator

        scenario_generator = RandomScenarioGenerator(
            max_num_jobs=3, max_num_ops=3, max_op_duration=5
        )
        schedule_generator = ScheduleGenerator(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        return JobShop(
            scenario_generator=scenario_generator,
            schedule_generator=schedule_generator,
            time_limit=10,
            neighborhood=Neighborhood.N5,
            scheduling_method=SchedulingMethod.FLOW_DUE_DATE_MOST_WORK,
        )

    @pytest.fixture
    def initial_state_and_timestep(
        self, step_job_shop_env: JobShop
    ) -> Tuple[ImprovementState, TimeStep[Observation]]:
        """Get initial state and timestep."""
        key = jax.random.PRNGKey(0)
        return step_job_shop_env.reset(key)

    def test_step_reward_types(self, step_job_shop_env: JobShop) -> None:
        """Test different reward types."""
        key = jax.random.PRNGKey(0)

        # Test incumbent reward
        env_incumbent = JobShop(
            scenario_generator=step_job_shop_env.scenario_generator,
            schedule_generator=step_job_shop_env.schedule_generator,
            reward_type="incumbent",
            time_limit=5,
        )
        state, _ = env_incumbent.reset(key)
        if jnp.any(state.action_mask):
            valid_actions = jnp.where(state.action_mask)
            action = jnp.array([valid_actions[0][0], valid_actions[1][0]], dtype=jnp.int32)
            _, timestep = env_incumbent.step(state, action)
            assert timestep.reward >= 0  # Incumbent reward should be non-negative

        # Test composed reward
        env_composed = JobShop(
            scenario_generator=step_job_shop_env.scenario_generator,
            schedule_generator=step_job_shop_env.schedule_generator,
            reward_type="composed",
            reward_scale=0.5,
            time_limit=5,
        )
        state, _ = env_composed.reset(key)
        if jnp.any(state.action_mask):
            valid_actions = jnp.where(state.action_mask)
            action = jnp.array([valid_actions[0][0], valid_actions[1][0]], dtype=jnp.int32)
            _, timestep = env_composed.step(state, action)
            # Composed reward can be positive or negative
            assert jnp.isfinite(timestep.reward)

        # Test local reward
        env_local = JobShop(
            scenario_generator=step_job_shop_env.scenario_generator,
            schedule_generator=step_job_shop_env.schedule_generator,
            reward_type="local",
            time_limit=5,
        )
        state, _ = env_local.reset(key)
        if jnp.any(state.action_mask):
            valid_actions = jnp.where(state.action_mask)
            action = jnp.array([valid_actions[0][0], valid_actions[1][0]], dtype=jnp.int32)
            _, timestep = env_local.step(state, action)
            # Local reward can be positive or negative
            assert jnp.isfinite(timestep.reward)

    def test_step_termination_conditions(self, step_job_shop_env: JobShop) -> None:
        """Test step termination conditions."""
        key = jax.random.PRNGKey(0)
        state, _ = step_job_shop_env.reset(key)

        # Test time limit truncation
        state = state.replace(step_count=step_job_shop_env.time_limit)  # type: ignore
        if jnp.any(state.action_mask):
            valid_actions = jnp.where(state.action_mask)
            action = jnp.array([valid_actions[0][0], valid_actions[1][0]], dtype=jnp.int32)
            _, timestep = step_job_shop_env.step(state, action)
            assert timestep.last()  # Should be LAST timestep
            assert timestep.discount == 1  # Truncation: discount = 1


class TestJobShopRestartFromBestComprehensive:
    """Comprehensive test suite for restart from best functionality."""

    def test_restart_from_best_functionality(self) -> None:
        """Test restart from best initialization and solution tracking."""
        from jumanji.environments.packing.job_shop.improvement.generator import ScheduleGenerator

        scenario_generator = RandomScenarioGenerator(
            max_num_jobs=3, max_num_ops=3, max_op_duration=5
        )
        schedule_generator = ScheduleGenerator(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        restart_env = JobShop(
            scenario_generator=scenario_generator,
            schedule_generator=schedule_generator,
            restart_from_best=True,
            nb_steps_before_restart=3,
            time_limit=10,
        )

        # Test initialization
        assert restart_env.restart_from_best is True
        assert restart_env.nb_steps_before_restart == 3

        key = jax.random.PRNGKey(0)
        state, _ = restart_env.reset(key)

        # Test initial state
        assert state.step_since_best == 0
        assert state.best_solution_so_far is not None

        # Test best solution tracking
        initial_best = state.best_solution_so_far

        # Best solution should match initial state
        assert jnp.allclose(initial_best.scheduled_times, state.scheduled_times)
        assert initial_best.adj_mat_pc.shape == state.adj_mat_pc.shape
        assert initial_best.adj_mat_mc.shape == state.adj_mat_mc.shape

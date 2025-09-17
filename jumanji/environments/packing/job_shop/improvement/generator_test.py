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


import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.packing.job_shop.conftest import DummyScenarioGenerator
from jumanji.environments.packing.job_shop.improvement.conftest import DummyScheduleGenerator
from jumanji.environments.packing.job_shop.improvement.generator import (
    RandomScheduleGenerator,
)
from jumanji.environments.packing.job_shop.improvement.types import ImprovementState, Neighborhood
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestMatrixGenerator:
    def test_init_adj_mat_pc(
        self,
        dummy_scenario_generator: DummyScenarioGenerator,
        dummy_schedule_generator: DummyScheduleGenerator,
    ) -> None:
        """Test that the precedence constraint matrix is correctly initialized."""

        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        adj_mat_pc = dummy_schedule_generator.init_adj_mat_pc(
            scenario.ops_durations, scenario.num_ops_per_job
        )

        assert adj_mat_pc.shape == (
            scenario.max_num_jobs * scenario.max_num_ops + 2,
            scenario.max_num_jobs * scenario.max_num_ops + 2,
        )

        num_source_links = jnp.sum(adj_mat_pc[0] == 1)
        assert num_source_links == scenario.num_jobs

        num_target_links = jnp.sum(adj_mat_pc[:, -1] > 0)
        assert num_target_links == scenario.num_jobs

        num_total_ops_links = jnp.sum(adj_mat_pc[1:, 1:] > 0)
        assert num_total_ops_links == jnp.sum(scenario.num_ops_per_job)

    def test_init_adj_mat_pc__jit(
        self,
        dummy_schedule_generator: DummyScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the precedence constraint matrix is correctly initialized,
        that the jit function is compiled only once."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_schedule_generator.init_adj_mat_pc, n=1))
        adj_mat_pc = call_fn(scenario.ops_durations, scenario.num_ops_per_job)
        adj_mat_pc2 = call_fn(scenario.ops_durations, scenario.num_ops_per_job)
        assert jnp.all(adj_mat_pc == adj_mat_pc2)


class TestDummyGenerator:
    def test_dummy_generator__properties(
        self, dummy_schedule_generator: DummyScheduleGenerator
    ) -> None:
        """Validate that the dummy instance generator has the correct properties."""
        assert dummy_schedule_generator.num_jobs == 3
        assert dummy_schedule_generator.num_machines == 3
        assert dummy_schedule_generator.max_num_ops == 3

    def test_dummy_generator__call(
        self,
        dummy_schedule_generator: DummyScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Check the scheduled times of the solution"""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)
        state = dummy_schedule_generator(key, scenario, method_id=0)
        scheduled_times = state.scheduled_times
        assert scheduled_times.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert jnp.all(
            scheduled_times
            == jnp.array(
                [
                    [0, 3, 6],
                    [3, 5, 9],
                    [5, 9, -1],
                ],
                jnp.int32,
            )
        )
        assert state.makespan == 13

    def test_dummy_generator__call_jit(
        self,
        dummy_schedule_generator: DummyScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Validate that the dummy instance generator's call function behaves correctly,
        that it is jit-table and compiles only once, and that it returns the same state
        for different keys.
        """
        scenario = dummy_scenario_generator(jax.random.PRNGKey(0), num_jobs=3, num_machines=3)

        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_schedule_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1), scenario, method_id=0, neighborhood=Neighborhood.N5)
        state2 = call_fn(jax.random.PRNGKey(2), scenario, method_id=0, neighborhood=Neighborhood.N5)
        assert_trees_are_equal(state1, state2)


class TestRandomGenerator:
    @pytest.fixture
    def random_schedule_generator(self) -> RandomScheduleGenerator:
        return RandomScheduleGenerator(
            num_jobs=3,
            num_machines=3,
            max_num_jobs=3,
            max_num_ops=3,
        )

    def test_random_generator__properties(
        self, random_schedule_generator: RandomScheduleGenerator
    ) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert random_schedule_generator.num_jobs == 3
        assert random_schedule_generator.num_machines == 3
        assert random_schedule_generator.max_num_ops == 3

    def test_init_adj_mat_mc_with_plist(
        self,
        dummy_scenario_generator: DummyScenarioGenerator,
        random_schedule_generator: RandomScheduleGenerator,
    ) -> None:
        """Test that the machine constraint matrix is correctly initialized."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        adj_mat_mc = random_schedule_generator.init_adj_mat_mc_with_plist(
            key, scenario.ops_machine_ids, scenario.ops_durations
        )

        assert adj_mat_mc.shape == (
            scenario.max_num_jobs * scenario.max_num_ops + 2,
            scenario.max_num_jobs * scenario.max_num_ops + 2,
        )

        # Check that source and target nodes are not connected to anything
        assert jnp.sum(adj_mat_mc[0]) == 0
        assert jnp.sum(adj_mat_mc[-1]) == 0

        # Check that each operation is connected to at most one machine
        assert jnp.sum(adj_mat_mc > 0, axis=1).max() <= 1

        # With key=2, the matrix should be:
        assert jnp.all(
            adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

    def test_init_adj_mat_mc_with_plist__jit(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the machine constraint matrix is correctly initialized,
        that the jit function is compiled only once.
        Also check that it returns different matrices for different keys."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        chex.clear_trace_counter()
        call_fn = jax.jit(
            chex.assert_max_traces(random_schedule_generator.init_adj_mat_mc_with_plist, n=1)
        )
        adj_mat_mc1 = call_fn(key1, scenario.ops_machine_ids, scenario.ops_durations)
        adj_mat_mc2 = call_fn(key2, scenario.ops_machine_ids, scenario.ops_durations)

        assert adj_mat_mc1.shape == (
            scenario.max_num_jobs * scenario.max_num_ops + 2,
            scenario.max_num_jobs * scenario.max_num_ops + 2,
        )
        assert adj_mat_mc2.shape == (
            scenario.max_num_jobs * scenario.max_num_ops + 2,
            scenario.max_num_jobs * scenario.max_num_ops + 2,
        )
        assert jnp.any(adj_mat_mc1 != adj_mat_mc2)

    def test_random_generator__init_adj_mat_mc_spt(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the machine constraint matrix is correctly initialized
        when using shortest processing time rule."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        # Test with shortest processing time rule
        adj_mat_mc = random_schedule_generator.init_adj_mat_mc_with_spt(
            scenario.ops_machine_ids, scenario.ops_durations
        )

        # Check output shape
        expected_shape = (
            scenario.max_num_jobs * scenario.max_num_ops + 2,
            scenario.max_num_jobs * scenario.max_num_ops + 2,
        )
        assert adj_mat_mc.shape == expected_shape

        jax.debug.print("adj_mat_mc: {0}", adj_mat_mc)
        assert jnp.all(
            adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

    def test_random_generator__init_adj_mat_mc_spt__jit(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the machine constraint matrix is correctly initialized when using
        shortest processing time rule, that the jit function is compiled only once.
        Also check that it returns same matrices for different keys."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        chex.clear_trace_counter()
        call_fn = jax.jit(
            chex.assert_max_traces(random_schedule_generator.init_adj_mat_mc_with_spt, n=1)
        )
        adj_mat_mc1 = call_fn(scenario.ops_machine_ids, scenario.ops_durations)
        adj_mat_mc2 = call_fn(scenario.ops_machine_ids, scenario.ops_durations)
        assert jnp.all(adj_mat_mc1 == adj_mat_mc2)
        assert_trees_are_equal(adj_mat_mc1, adj_mat_mc2)

    def test_random_generator__best_solution_matches_state_initially(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """best_solution_so_far mirrors the initially generated state."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)
        for method_id in range(3):
            state = random_schedule_generator(
                key, scenario, method_id=method_id, neighborhood=Neighborhood.N5
            )
            best = state.best_solution_so_far
            assert jnp.all(best.scheduled_times == state.scheduled_times)
            assert jnp.all(best.adj_mat_pc == state.adj_mat_pc)
            assert jnp.all(best.adj_mat_mc == state.adj_mat_mc)
            assert jnp.all(best.action_mask == state.action_mask)
            assert jnp.all(best.critical_block_info == state.critical_block_info)
            assert jnp.all(best.gap_left_right == state.gap_left_right)
            assert jnp.all(best.est == state.est)
            assert jnp.all(best.lst == state.lst)

    def test_random_generator__call(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the random instance generator's call function behaves correctly."""
        key = jax.random.PRNGKey(0)  # with this key, job priority list is [0, 1, 2]
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        state1 = random_schedule_generator(key, scenario, method_id=0, neighborhood=Neighborhood.N5)
        assert state1.ops_machine_ids.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state1.ops_durations.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state1.num_ops_per_job.shape == (scenario.num_jobs,)

        # Given key=2, the priority list is [0, 1, 2], scheduled times of tasks should
        # be the following:
        jax.debug.print("state1.scheduled_times: {0}", state1.scheduled_times)
        assert jnp.all(
            state1.scheduled_times
            == jnp.array(
                [
                    [0, 3, 5],
                    [3, 7, 8],
                    [12, 16, -1],  # last operation is not scheduled
                ]
            )
        )

        state2 = random_schedule_generator(key, scenario, method_id=1, neighborhood=Neighborhood.N5)
        assert state2.ops_machine_ids.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state2.ops_durations.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state2.num_ops_per_job.shape == (scenario.num_jobs,)

        # Given the shortest processing time rule, scheduled times of tasks should be the following:
        assert jnp.all(
            state2.scheduled_times
            == jnp.array(
                [
                    [2, 5, 7],
                    [0, 2, 7],
                    [11, 15, -1],  # last operation is not scheduled
                ]
            )
        )

        state3 = random_schedule_generator(key, scenario, method_id=2, neighborhood=Neighborhood.N5)
        assert jnp.all(
            state3.scheduled_times
            == jnp.array(
                [
                    [2, 5, 7],
                    [0, 2, 7],
                    [0, 4, -1],  # last operation is not scheduled
                ]
            )
        )

    def test_random_generator__call_jit(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Validate that the random instance generator's call function is jit-able and compiles
        only once. Also check that giving two different keys results in two different instances.
        Do this for all method_ids (0, 1, 2).
        """
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_schedule_generator.__call__, n=1))

        # Method 0 includes stochasticity, so different keys should result in different schedules.
        state1 = call_fn(
            key=jax.random.PRNGKey(1), scenario=scenario, method_id=0, neighborhood=Neighborhood.N5
        )
        assert isinstance(state1, ImprovementState)

        state2 = call_fn(
            key=jax.random.PRNGKey(2), scenario=scenario, method_id=0, neighborhood=Neighborhood.N5
        )
        assert_trees_are_different(state1.scheduled_times, state2.scheduled_times)

        # Check that the call function compiles only once for each method_id.
        for method_id in range(3):
            chex.clear_trace_counter()
            call_fn = jax.jit(chex.assert_max_traces(random_schedule_generator.__call__, n=1))
            state1 = call_fn(
                key=jax.random.PRNGKey(1),
                scenario=scenario,
                method_id=method_id,
                neighborhood=Neighborhood.N5,
            )
            assert isinstance(state1, ImprovementState)

            state2 = call_fn(
                key=jax.random.PRNGKey(1),
                scenario=scenario,
                method_id=method_id,
                neighborhood=Neighborhood.N5,
            )
            assert_trees_are_equal(state1, state2)

    def test_random_generator__init_adj_mat_mc_fdd_mwr(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the machine constraint matrix is correctly initialized when using
        FDD-MWR rule.
        """
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        state = random_schedule_generator(key, scenario, method_id=2, neighborhood=Neighborhood.N6)
        assert state.ops_machine_ids.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state.ops_durations.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state.num_ops_per_job.shape == (scenario.num_jobs,)

        call_fn = jax.jit(chex.assert_max_traces(random_schedule_generator.__call__, n=1))
        state1 = call_fn(
            key=jax.random.PRNGKey(1), scenario=scenario, method_id=0, neighborhood=Neighborhood.N6
        )
        assert isinstance(state1, ImprovementState)

        state2 = call_fn(
            key=jax.random.PRNGKey(2), scenario=scenario, method_id=1, neighborhood=Neighborhood.N6
        )
        assert_trees_are_different(state1, state2)

    def test_random_generator__init_adj_mat_mc_fdd_mwr__jit(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the machine constraint matrix is correctly initialized when using
        FDD-MWR rule, that the jit function is compiled only once.
        Also check that it returns same matrices for different keys."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_schedule_generator.__call__, n=1))
        state1 = call_fn(
            key=jax.random.PRNGKey(1), scenario=scenario, method_id=2, neighborhood=Neighborhood.N6
        )
        assert isinstance(state1, ImprovementState)

        state2 = call_fn(
            key=jax.random.PRNGKey(2), scenario=scenario, method_id=2, neighborhood=Neighborhood.N6
        )

        assert jnp.all(state1.scheduled_times == state2.scheduled_times)

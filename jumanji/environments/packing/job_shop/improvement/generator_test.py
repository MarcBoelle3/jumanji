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
from jumanji.environments.packing.job_shop.improvement.types import ImprovementState
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
        jax.debug.print("scheduled_times: {0}", scheduled_times)
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
        state1 = call_fn(jax.random.PRNGKey(1), scenario)
        state2 = call_fn(jax.random.PRNGKey(2), scenario)
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

        plist = jnp.arange(scenario.max_num_jobs)
        adj_mat_mc = random_schedule_generator.init_adj_mat_mc_with_plist(
            plist, scenario.ops_machine_ids, scenario.ops_durations
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
        plist1 = jnp.arange(scenario.max_num_jobs)
        plist2 = jnp.arange(scenario.max_num_jobs)[::-1]
        chex.clear_trace_counter()
        call_fn = jax.jit(
            chex.assert_max_traces(random_schedule_generator.init_adj_mat_mc_with_plist, n=1)
        )
        adj_mat_mc1 = call_fn(plist1, scenario.ops_machine_ids, scenario.ops_durations)
        adj_mat_mc2 = call_fn(plist2, scenario.ops_machine_ids, scenario.ops_durations)

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

        # Check that matrix is properly initialized
        assert jnp.all(
            adj_mat_mc
            == jnp.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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

    def test_random_generator__call(
        self,
        random_schedule_generator: RandomScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that the random instance generator's call function behaves correctly."""
        key = jax.random.PRNGKey(0)  # with this key, job priority list is [0, 1, 2]
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        state1 = random_schedule_generator(key, scenario, method_id=0)
        assert state1.ops_machine_ids.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state1.ops_durations.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state1.num_ops_per_job.shape == (scenario.num_jobs,)

        # Given the priority list [0, 1, 2], scheduled times of tasks should be the following:
        assert jnp.all(
            state1.scheduled_times
            == jnp.array(
                [
                    [0, 3, 5],
                    [3, 7, 8],
                    [12, 16, -jnp.inf],  # last operation is not scheduled
                ]
            )
        )

        state2 = random_schedule_generator(key, scenario, method_id=1)
        assert state2.ops_machine_ids.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state2.ops_durations.shape == (scenario.num_jobs, scenario.max_num_ops)
        assert state2.num_ops_per_job.shape == (scenario.num_jobs,)

        # Given the shortest processing time rule, scheduled times of tasks should be the following:
        jax.debug.print("state2.scheduled_times: {0}", state2.scheduled_times)
        assert jnp.all(
            state2.scheduled_times
            == jnp.array(
                [
                    [2, 5, 7],
                    [0, 2, 7],
                    [11, 15, -jnp.inf],  # last operation is not scheduled
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
        """
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(random_schedule_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario, method_id=0)
        assert isinstance(state1, ImprovementState)

        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario, method_id=1)
        assert_trees_are_different(state1, state2)

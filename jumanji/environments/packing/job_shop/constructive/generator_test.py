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
import pytest

from jumanji.environments.packing.job_shop.conftest import DummyScenarioGenerator
from jumanji.environments.packing.job_shop.constructive.conftest import DummyScheduleGenerator
from jumanji.environments.packing.job_shop.constructive.generator import (
    EmptyScheduleGenerator,
    ToyScheduleGenerator,
)
from jumanji.environments.packing.job_shop.constructive.types import ConstructiveState
from jumanji.environments.packing.job_shop.scenario_generator import (
    RandomScenarioGenerator,
    ToyScenarioGenerator,
)
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestDummyScheduleGenerator:
    def test_dummy_schedule_generator__properties(
        self, dummy_schedule_generator: DummyScheduleGenerator
    ) -> None:
        """Validate that the dummy schedule generator has the correct properties."""
        assert dummy_schedule_generator.num_jobs == 3
        assert dummy_schedule_generator.num_machines == 3
        assert dummy_schedule_generator.max_num_ops == 3

    def test_dummy_schedule_generator__call(
        self,
        dummy_schedule_generator: DummyScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Validate that the dummy schedule generator's call function behaves correctly,
        that it is jit-table and compiles only once, and that it returns the same state
        for different keys.
        """
        scenario = dummy_scenario_generator(key=jax.random.PRNGKey(1), num_jobs=3, num_machines=3)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_schedule_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario)
        assert_trees_are_equal(state1, state2)


class TestToyScheduleGenerator:
    @pytest.fixture
    def toy_schedule_generator(self) -> ToyScheduleGenerator:
        return ToyScheduleGenerator()

    @pytest.fixture
    def toy_scenario_generator(self) -> ToyScenarioGenerator:
        return ToyScenarioGenerator()

    def test_toy_schedule_generator__properties(
        self, toy_schedule_generator: ToyScheduleGenerator
    ) -> None:
        """Validate that the toy schedule generator has the correct properties."""
        assert toy_schedule_generator.num_jobs == 5
        assert toy_schedule_generator.num_machines == 4
        assert toy_schedule_generator.max_num_ops == 4

    def test_toy_schedule_generator__call(
        self,
        toy_schedule_generator: ToyScheduleGenerator,
        toy_scenario_generator: ToyScenarioGenerator,
    ) -> None:
        """Validate that the toy schedule generator's call function behaves correctly,
        that it is jit-able and compiles only once, and that it returns the same state
        for different keys.
        """
        scenario = toy_scenario_generator(key=jax.random.PRNGKey(1), num_jobs=5, num_machines=4)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_schedule_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario)
        assert_trees_are_equal(state1, state2)


class TestEmptyScheduleGenerator:
    @pytest.fixture
    def empty_schedule_generator(self) -> EmptyScheduleGenerator:
        return EmptyScheduleGenerator(
            num_jobs=20,
            num_machines=10,
            max_num_ops=15,
        )

    def test_empty_schedule_generator__properties(
        self, empty_schedule_generator: EmptyScheduleGenerator
    ) -> None:
        """Validate that the random schedule generator has the correct properties."""
        assert empty_schedule_generator.num_jobs == 20
        assert empty_schedule_generator.num_machines == 10
        assert empty_schedule_generator.max_num_ops == 15

    def test_empty_schedule_generator__call(
        self,
        empty_schedule_generator: EmptyScheduleGenerator,
        random_scenario_generator: RandomScenarioGenerator,
    ) -> None:
        """Validate that the random schedule generator's call function is jit-able and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        scenario = random_scenario_generator(
            key=jax.random.PRNGKey(1), num_jobs=20, num_machines=10
        )

        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(empty_schedule_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        assert isinstance(state1, ConstructiveState)

        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario)
        assert_trees_are_different(state1, state2)

    def test_empty_schedule_generator__call_same_key(
        self,
        empty_schedule_generator: EmptyScheduleGenerator,
        random_scenario_generator: RandomScenarioGenerator,
    ) -> None:
        """Validate that the random schedule generator's call function is jit-able and compiles
        only once. Also check that giving the same key results in the same instance.
        """
        scenario = random_scenario_generator(
            key=jax.random.PRNGKey(1), num_jobs=20, num_machines=10
        )

        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(empty_schedule_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        state2 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        assert_trees_are_equal(state1, state2)

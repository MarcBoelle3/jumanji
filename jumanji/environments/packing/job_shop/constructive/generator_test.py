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

from jumanji.environments.packing.job_shop.constructive.conftest import DummyGenerator
from jumanji.environments.packing.job_shop.constructive.generator import (
    StandardGenerator,
    ToyGenerator,
)
from jumanji.environments.packing.job_shop.constructive.types import State
from jumanji.environments.packing.job_shop.scenario_generator import RandomScenarioGenerator
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestDummyGenerator:
    @pytest.fixture
    def dummy_generator(self) -> DummyGenerator:
        return DummyGenerator()

    def test_dummy_generator__properties(self, dummy_generator: DummyGenerator) -> None:
        """Validate that the dummy instance generator has the correct properties."""
        assert dummy_generator.num_jobs == 3
        assert dummy_generator.num_machines == 3
        assert dummy_generator.max_num_ops == 3

    def test_dummy_generator__call(
        self, dummy_generator: DummyGenerator, dummy_scenario_generator: RandomScenarioGenerator
    ) -> None:
        """Validate that the dummy instance generator's call function behaves correctly,
        that it is jit-table and compiles only once, and that it returns the same state
        for different keys.
        """
        scenario = dummy_scenario_generator(key=jax.random.PRNGKey(1), num_jobs=3, num_machines=3)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario)
        assert_trees_are_equal(state1, state2)


class TestToyGenerator:
    @pytest.fixture
    def toy_generator(self) -> ToyGenerator:
        return ToyGenerator()

    def test_toy_generator__properties(self, toy_generator: ToyGenerator) -> None:
        """Validate that the toy instance generator has the correct properties."""
        assert toy_generator.num_jobs == 5
        assert toy_generator.num_machines == 4
        assert toy_generator.max_num_ops == 4

    def test_toy_generator__call(
        self, toy_generator: ToyGenerator, dummy_scenario_generator: RandomScenarioGenerator
    ) -> None:
        """Validate that the toy instance generator's call function behaves correctly,
        that it is jit-able and compiles only once, and that it returns the same state
        for different keys.
        """
        scenario = dummy_scenario_generator(key=jax.random.PRNGKey(1), num_jobs=5, num_machines=4)
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(toy_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario)
        assert_trees_are_equal(state1, state2)


class TestStandardGenerator:
    @pytest.fixture
    def standard_generator(self) -> StandardGenerator:
        return StandardGenerator(
            num_jobs=20,
            num_machines=10,
            max_num_ops=15,
        )

    def test_standard_generator__properties(self, standard_generator: StandardGenerator) -> None:
        """Validate that the random instance generator has the correct properties."""
        assert standard_generator.num_jobs == 20
        assert standard_generator.num_machines == 10
        assert standard_generator.max_num_ops == 15

    def test_standard_generator__call(
        self,
        standard_generator: StandardGenerator,
        dummy_scenario_generator: RandomScenarioGenerator,
    ) -> None:
        """Validate that the random instance generator's call function is jit-able and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        scenario = dummy_scenario_generator(key=jax.random.PRNGKey(1), num_jobs=20, num_machines=10)

        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(standard_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1), scenario=scenario)
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2), scenario=scenario)
        assert_trees_are_different(state1, state2)

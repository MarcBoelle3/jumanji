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
import jax.numpy as jnp
import pytest

from jumanji.environments.packing.job_shop.scenario_generator import (
    RandomScenarioGenerator,
    ScenarioGenerator,
)
from jumanji.environments.packing.job_shop.types import Scenario


class DummyScenarioGenerator(ScenarioGenerator):
    """Hardcoded `ScenarioGenerator` mainly used for testing and debugging. It deterministically
    outputs a hardcoded instance with 3 jobs, 3 machines, a max of 3 ops for any job, and a max
    duration of 4 time steps for any operation.
    """

    def __init__(self) -> None:
        super().__init__(max_num_jobs=3, max_num_ops=3, max_op_duration=4)

    def __call__(self, key: chex.PRNGKey, num_jobs: int, num_machines: int) -> Scenario:
        """Call method responsible for generating a new scenario. It returns a hardcoded instance
        with 3 jobs, 3 machines, a max of 3 ops for any job, and a max duration of 4 time steps
        for any operation.
        """
        del key
        del num_jobs
        del num_machines

        ops_machine_ids = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, -1],
            ],
            jnp.int32,
        )
        ops_durations = jnp.array(
            [
                [3, 2, 2],
                [2, 1, 4],
                [4, 3, -1],
            ],
            jnp.int32,
        )

        return Scenario(
            num_jobs=3,
            num_machines=3,
            max_num_jobs=3,
            max_num_ops=3,
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            num_ops_per_job=jnp.array([3, 3, 2], jnp.int32),
        )


@pytest.fixture
def dummy_scenario_generator() -> DummyScenarioGenerator:
    """Create a dummy RandomScenarioGenerator with fixed parameters."""
    return DummyScenarioGenerator()


@pytest.fixture
def random_scenario_generator() -> RandomScenarioGenerator:
    """Create a random RandomScenarioGenerator with fixed parameters."""
    max_num_jobs = 5
    max_num_ops = 3
    max_op_duration = 4
    return RandomScenarioGenerator(max_num_jobs, max_num_ops, max_op_duration)

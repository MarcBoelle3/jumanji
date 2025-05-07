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

from jumanji.environments.packing.job_shop.scenario_generator import (
    RandomScenarioGenerator,
)
from jumanji.environments.packing.job_shop.types import Scenario
from jumanji.testing.pytrees import assert_trees_are_different


class TestRandomScenarioGenerator:
    def test_random_scenario_generator__properties(
        self, dummy_scenario_generator: RandomScenarioGenerator
    ) -> None:
        """Test that the properties of the RandomScenarioGenerator are correct."""
        assert dummy_scenario_generator.max_num_jobs == 5
        assert dummy_scenario_generator.max_num_ops == 3
        assert dummy_scenario_generator.max_op_duration == 4

    def test_random_instance_generator_output_shapes(
        self, dummy_scenario_generator: RandomScenarioGenerator
    ) -> None:
        """Test that random_instance_generator returns arrays of the expected shapes."""

        num_jobs = 3
        num_machines = 4

        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs, num_machines)

        # === Check shapes ===

        assert scenario.ops_machine_ids.shape == (
            dummy_scenario_generator.max_num_jobs,
            dummy_scenario_generator.max_num_ops,
        )
        assert scenario.ops_durations.shape == (
            dummy_scenario_generator.max_num_jobs,
            dummy_scenario_generator.max_num_ops,
        )
        assert scenario.num_ops_per_job.shape == (dummy_scenario_generator.max_num_jobs,)

        # === Check valid values ===

        # Check valid machine IDs (-1 for padding or valid machine ID)
        assert jnp.all(
            (scenario.ops_machine_ids == -1)
            | ((scenario.ops_machine_ids >= 0) & (scenario.ops_machine_ids < num_machines))
        )

        # Check valid durations (-1 for padding or positive duration <= max_op_duration)
        assert jnp.all(
            (scenario.ops_durations == -1)
            | (
                (scenario.ops_durations > 0)
                & (scenario.ops_durations <= dummy_scenario_generator.max_op_duration)
            )
        )

        # Check valid number of ops per job (between 1 and max_num_ops)
        assert jnp.all(
            (scenario.num_ops_per_job >= 0)
            & (scenario.num_ops_per_job <= dummy_scenario_generator.max_num_ops)
        )

        # Check valid number of jobs (equal to num_jobs)
        assert jnp.sum(jnp.any(scenario.ops_machine_ids != -1, axis=1)) == num_jobs

        # === Check padding ===

        assert jnp.all(scenario.ops_machine_ids[num_jobs:] == -1)
        assert jnp.all(scenario.ops_durations[num_jobs:] == -1)
        assert jnp.all(scenario.num_ops_per_job[num_jobs:] == 0)

    def test_random_instance_generator_different_keys(
        self, dummy_scenario_generator: RandomScenarioGenerator
    ) -> None:
        """Validate that the random instance generator's call function is jit-able and compiles
        only once. Also check that giving two different keys results in two different instances.
        """
        num_jobs = 3
        num_machines = 4

        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_scenario_generator.__call__, n=1))
        scenario1 = call_fn(jax.random.PRNGKey(1), num_jobs, num_machines)
        assert isinstance(scenario1, Scenario)
        scenario2 = call_fn(jax.random.PRNGKey(2), num_jobs, num_machines)
        assert_trees_are_different(scenario1, scenario2)

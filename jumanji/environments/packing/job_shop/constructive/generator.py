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

from jumanji.environments.packing.job_shop.constructive.types import State
from jumanji.environments.packing.job_shop.types import Scenario


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class. A `Generator` is responsible
    for generating a problem instance when the environment is reset.
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_ops: int,
    ):
        """Abstract class implementing the attributes `num_jobs`, `num_machines`, `max_num_ops`.

        Args:
            num_jobs: the number of jobs that need to be scheduled.
            num_machines: the number of machines that the jobs can be scheduled on.
            max_num_ops: the maximum number of operations for any given job.
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_num_ops = max_num_ops

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, scenario: Scenario) -> State:
        """Call method responsible for generating a new state based on a given scenario.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.
            scenario: a `Scenario` object containing the problem instance.
        Returns:
            A `JobShop` environment state.
        """


class ToyGenerator(Generator):
    """`Generator` that can be used as an example. It deterministically outputs a hardcoded
    instance with 5 jobs, 4 machines, a max of 4 ops for any job, and max duration of 4 time
    steps for any operation. By construction, this generator has a known, optimal makespan
    of 8 time steps.
    """

    def __init__(self) -> None:
        super().__init__(num_jobs=5, num_machines=4, max_num_ops=4)

    def __call__(self, key: chex.PRNGKey, scenario: Scenario) -> State:
        del scenario
        del key

        ops_machine_ids = jnp.array(
            [
                [2, 3, 1, 2],
                [3, 2, 0, -1],
                [1, 3, -1, -1],
                [0, 3, 0, 0],
                [1, 0, 1, -1],
            ],
            jnp.int32,
        )
        ops_durations = jnp.array(
            [
                [2, 2, 1, 2],
                [2, 4, 1, -1],
                [2, 3, -1, -1],
                [4, 1, 1, 1],
                [3, 1, 2, -1],
            ],
            jnp.int32,
        )

        # Initially, all machines are available (the value self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.full(self.num_machines, self.num_jobs, jnp.int32)
        machines_remaining_times = jnp.full(self.num_machines, 0, jnp.int32)
        scheduled_times = jnp.full((self.num_jobs, self.max_num_ops), -1, jnp.int32)
        ops_mask = ops_machine_ids != -1
        step_count = jnp.array(0, jnp.int32)

        state = State(
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=None,
            step_count=step_count,
            scheduled_times=scheduled_times,
            key=jax.random.PRNGKey(0),
        )

        return state


class EmptyScheduleGenerator(Generator):
    """Instance generator that initializes the state based on a given scenario.
    All machines are available at the beginning of the episode, and the scheduled times are
    initialized to -1 (no operation scheduled yet).
    """

    def __init__(self, num_jobs: int, num_machines: int, max_num_ops: int):
        super().__init__(num_jobs, num_machines, max_num_ops)

    def __call__(self, key: chex.PRNGKey, scenario: Scenario) -> State:
        # Generate a random scenario
        ops_machine_ids = scenario.ops_machine_ids
        ops_durations = scenario.ops_durations

        # Initially, all machines are available (the value self.num_jobs corresponds to no-op)
        machines_job_ids = jnp.full(self.num_machines, self.num_jobs, jnp.int32)
        machines_remaining_times = jnp.full(self.num_machines, 0, jnp.int32)

        # Initially, none of the operations have been scheduled
        scheduled_times = jnp.full((self.num_jobs, self.max_num_ops), -1, jnp.int32)
        ops_mask = ops_machine_ids != -1

        # Time starts at 0
        step_count = jnp.array(0, jnp.int32)

        state = State(
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            ops_mask=ops_mask,
            machines_job_ids=machines_job_ids,
            machines_remaining_times=machines_remaining_times,
            action_mask=None,
            step_count=step_count,
            scheduled_times=scheduled_times,
            key=key,
        )

        return state

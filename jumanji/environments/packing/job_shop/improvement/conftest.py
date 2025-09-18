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


import jax.numpy as jnp
import jax.random
import pytest
from chex import PRNGKey

from jumanji.environments.packing.job_shop.improvement.generator import ScheduleGenerator
from jumanji.environments.packing.job_shop.improvement.types import (
    ImprovementState,
    Neighborhood,
    SchedulingMethod,
)
from jumanji.environments.packing.job_shop.types import Scenario


class DummyScheduleGenerator(ScheduleGenerator):
    """Hardcoded `Generator` mainly used for testing and debugging. It deterministically
    outputs a hardcoded instance with 3 jobs, 3 machines, a max of 3 ops for any job, and a max
    duration of 4 time steps for any operation.
    """

    def __init__(self) -> None:
        super().__init__(num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3)

    def __call__(
        self,
        key: PRNGKey,
        scenario: Scenario,
        method: SchedulingMethod,
        neighborhood: Neighborhood = Neighborhood.N5,
    ) -> ImprovementState:
        """Call method responsible for generating a new state. It returns a job shop scheduling
        instance with a non-optimal valid solution.

        Args:
            key: jax random key for any stochasticity used in the generation process. Not used
                in this generator.
            scenario: Scenario object containing the problem definition. Not used in this generator.
            method: Scheduling method enum. Not used in this generator.
            neighborhood: Neighborhood type (N5 or N6). Not used in this generator.
        Returns:
            A JobShop State.
        """
        del key
        del scenario
        del method
        del neighborhood

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
            jnp.float32,
        )

        scheduled_times = jnp.array(
            [
                [0, 3, 6],
                [3, 5, 9],
                [5, 9, -1],
            ],
            jnp.float32,
        )

        num_jobs, max_num_ops = ops_machine_ids.shape
        mask = ops_machine_ids != -1
        num_ops_per_job = jnp.sum(mask, axis=1)

        step_count = jnp.array(0, jnp.int32)

        adj_mat_pc = self.build_precedence_matrix(ops_durations, num_ops_per_job)
        adj_mat_mc = jnp.array(
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

        is_on_critical_path = jnp.array(
            [
                [True, True, False],
                [False, False, True],
                [True, False, False],
            ]
        )

        # N5 neighborhood
        action_mask = jnp.array(
            [
                [False, False],
                [False, True],
                [False, False],
                [False, False],
                [False, False],
                [True, False],
                [False, False],
                [False, False],
                [False, False],
            ],
            jnp.bool_,
        )

        # Critical block info: for each operation (row), it contains:
        # [is_critical, num_critical_block, is_left, is_right, left_neighbor, right_neighbor,
        # left_end_of_critical_block, right_end_of_critical_block]

        critical_block_info = jnp.array(
            [
                [1, 0, 1, 1, -1, -1, 0, -1],
                [1, 1, 1, 0, -1, 6, 1, 5],
                [0, 2, 1, 1, -1, -1, 2, -1],
                [0, 3, 1, 1, -1, -1, 3, -1],
                [0, 4, 1, 1, -1, -1, 4, -1],
                [1, 1, 0, 1, 6, -1, 1, 5],
                [1, 1, 0, 0, 1, 5, 1, 5],
                [0, 7, 1, 1, -1, -1, 7, -1],
                [0, 8, 1, 1, -1, -1, 8, -1],
            ]
        )

        # Add missing fields for ImprovementState
        gap_left_right = jnp.zeros((9, 2), dtype=jnp.float32)  # (max_num_jobs * max_num_ops, 2)
        est = jnp.array(
            [0, 0, 3, 6, 3, 5, 9, 5, 9, 0, 0], dtype=jnp.float32
        )  # (max_num_jobs * max_num_ops + 2,)
        lst = jnp.array(
            [0, 0, 3, 6, 3, 5, 9, 5, 9, 0, 0], dtype=jnp.float32
        )  # (max_num_jobs * max_num_ops + 2,)
        # Create a dummy best solution
        from jumanji.environments.packing.job_shop.improvement.types import BestSolution

        best_solution_so_far = BestSolution(
            scheduled_times=scheduled_times,
            adj_mat_pc=adj_mat_pc,
            adj_mat_mc=adj_mat_mc,
            is_on_critical_path=is_on_critical_path,
            action_mask=action_mask,
            critical_block_info=critical_block_info,
            gap_left_right=gap_left_right,
            est=est,
            lst=lst,
        )

        state = ImprovementState(
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            num_ops_per_job=num_ops_per_job,
            step_count=step_count,
            scheduled_times=scheduled_times,
            adj_mat_pc=adj_mat_pc,
            adj_mat_mc=adj_mat_mc,
            makespan=jnp.array(13, jnp.int32),
            incumbent_makespan=jnp.array(13, jnp.int32),
            step_minimum=jnp.array(0, jnp.int32),
            is_on_critical_path=is_on_critical_path,
            action_mask=action_mask,
            critical_block_info=critical_block_info,
            gap_left_right=gap_left_right,
            est=est,
            lst=lst,
            best_solution_so_far=best_solution_so_far,
            step_since_best=jnp.array(0, jnp.int32),
            key=jax.random.PRNGKey(0),
        )

        return state


@pytest.fixture
def dummy_schedule_generator() -> DummyScheduleGenerator:
    return DummyScheduleGenerator()

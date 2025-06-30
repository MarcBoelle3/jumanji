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

from jumanji.environments.packing.job_shop.types import Scenario


class ScenarioGenerator(abc.ABC):
    """Defines the abstract `ScenarioGenerator` base class. A `ScenarioGenerator` is responsible
    for generating a problem scenario. This includes initializing the number of jobs, machines,
    operations and their assignment to machines.
    """

    def __init__(self, max_num_jobs: int, max_num_ops: int, max_op_duration: int):
        """Initialize the scenario parameters with the given parameters.

        Args:
            max_num_jobs: Maximum number of jobs that can be scheduled.
            max_num_ops: Maximum number of operations per job.
            max_op_duration: Maximum duration of any operation.
        """

        self.max_num_jobs = max_num_jobs
        self.max_num_ops = max_num_ops
        self.max_op_duration = max_op_duration

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey, num_jobs: int, num_machines: int) -> Scenario:
        """Call method responsible for generating a new scenario.

        Args:
            key: jax random key in case stochasticity is used in the instance generation process.
            num_jobs: Number of jobs in the instance. It has to be between 1 and self.max_num_jobs.
            num_machines: Number of machines in the instance.

        Returns:
            A 'Scenario' object, containing:
              - ops_machine_ids: Array (max_num_jobs, max_num_ops) indicating the machine for
                each operation, with -1 indicating that the operation does not exist.
              - ops_durations: Array (max_num_jobs, max_num_ops) indicating the duration of
                each operation, with -1 indicating that the operation does not exist.
              - num_ops_per_job: Array (max_num_jobs) indicating number of operations per job,
                with 0 indicating that the job does not exist.
              - key: jax random key used to generate the scenario.
        """


class ToyScenarioGenerator(ScenarioGenerator):
    """`ScenarioGenerator` that can be used as an example. It deterministically outputs a hardcoded
    instance with 5 jobs, 4 machines, a max of 4 ops for any job, and max duration of 4 time
    steps for any operation. By construction, this generator has a known, optimal makespan
    of 8 time steps.
    """

    def __init__(self) -> None:
        super().__init__(max_num_jobs=5, max_num_ops=4, max_op_duration=4)

    def __call__(self, key: chex.PRNGKey, num_jobs: int, num_machines: int) -> Scenario:
        del key
        del num_jobs
        del num_machines

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

        scenario = Scenario(
            num_jobs=5,
            num_machines=4,
            max_num_jobs=5,
            max_num_ops=4,
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            num_ops_per_job=jnp.array([4, 4, 2, 4, 3], jnp.int32),
            key=jax.random.PRNGKey(0),
        )

        return scenario


class RandomScenarioGenerator(ScenarioGenerator):
    """Scenario generator that generates random instances of the job shop scheduling problem.
    Given the max number of machines, max number of jobs, max number of operations for any job,
    and max duration of any operation, the generation works as follows: for each job, we sample the
    number of ops for that job. Then, for each operation, a machine_id and duration are sampled,
    both from random uniform distributions. Finally, padding is done for jobs whose number of
    operations is less than the max.

    For instances with num_jobs lower than max_num_jobs, the extra jobs are padded with -1."""

    def __init__(self, max_num_jobs: int, max_num_ops: int, max_op_duration: int, hard_num_ops_per_job: bool = False) -> None:
        super().__init__(max_num_jobs, max_num_ops, max_op_duration)
        self.hard_num_ops_per_job = hard_num_ops_per_job

    def __call__(self, key: chex.PRNGKey, num_jobs: int, num_machines: int) -> Scenario:
        key, machine_key, duration_key, ops_key = jax.random.split(key, num=4)

        # Randomly sample machine IDs and durations
        ops_machine_ids = jax.random.randint(
            machine_key,
            shape=(self.max_num_jobs, self.max_num_ops),
            minval=0,
            maxval=num_machines,
        )
        ops_durations = jax.random.randint(
            duration_key,
            shape=(self.max_num_jobs, self.max_num_ops),
            minval=1,
            maxval=self.max_op_duration + 1,
        )

        # Vary the number of ops across jobs
        if self.hard_num_ops_per_job:
            num_ops_per_job = jnp.full((self.max_num_jobs,), self.max_num_ops, dtype=jnp.int32)
        else:
            num_ops_per_job = jax.random.randint(
                ops_key,
                shape=(self.max_num_jobs,),
                minval=1,
                maxval=self.max_num_ops + 1,
            )

        # Set number of jobs to 0 for non-existing jobs
        jobs_mask = jnp.less(jnp.arange(self.max_num_jobs), num_jobs)  # shape (max_num_jobs,)
        num_ops_per_job = jobs_mask * num_ops_per_job

        # Mask non-existing jobs and operations
        ops_mask = jnp.less(
            jnp.tile(jnp.arange(self.max_num_ops), reps=(self.max_num_jobs, 1)),
            jnp.expand_dims(num_ops_per_job, axis=-1),
        )

        total_mask = jnp.logical_and(ops_mask, jobs_mask[:, None])

        ops_machine_ids = jnp.where(total_mask, ops_machine_ids, jnp.array(-1, jnp.int32))
        ops_durations = jnp.where(total_mask, ops_durations, jnp.array(-1, jnp.int32))

        scenario = Scenario(
            num_jobs=num_jobs,
            num_machines=num_machines,
            max_num_jobs=self.max_num_jobs,
            max_num_ops=self.max_num_ops,
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            num_ops_per_job=num_ops_per_job,
            key=key,
        )

        return scenario


class FromDataSetScenarioGenerator(ScenarioGenerator):
    """Scenario generator that loads instances from a pre-existing dataset stored in a .npy file.
    The generator serves scenarios sequentially from the dataset on each call.
    """

    def __init__(self, dataset_path: str):
        """Initializes the generator by loading a dataset from a file.

        The dataset is expected to be a NumPy array with a specific shape.

        Args:
            dataset_path: Path to the .npy file containing the dataset. The array shape must be
                (n_instances, 2, n_jobs, n_machines), where the second dimension contains
                the duration matrix at index 0 and the machine assignment matrix at index 1.
        """
        try:
            dataset = jnp.load(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

        if dataset.ndim != 4 or dataset.shape[1] != 2:
            raise ValueError(
                f"Invalid dataset shape. Expected (n_instances, 2, n_jobs, n_machines), "
                f"but got {dataset.shape}."
            )

        self.dataset = dataset
        self.num_instances = dataset.shape[0]
        self.instance_idx = 0

        n_jobs = dataset.shape[2]
        n_machines = dataset.shape[3]  # In this formulation, n_machines is n_ops
        max_duration = jnp.max(self.dataset[:, 0, :, :])
        print(f"Type of max_duration: {type(max_duration)}")

        super().__init__(
            max_num_jobs=n_jobs,
            max_num_ops=n_machines,
            max_op_duration=int(max_duration),
        )

    def __call__(self, key: chex.PRNGKey, num_jobs: int, num_machines: int) -> Scenario:
        """Returns the next scenario from the loaded dataset.

        Ignores the input arguments as the scenarios are pre-determined by the dataset.

        Args:
            key: Unused jax random key.
            num_jobs: Unused number of jobs.
            num_machines: Unused number of machines.

        Returns:
            A `Scenario` object for the current instance.

        Raises:
            IndexError: If all instances from the dataset have already been served.
        """
        del key, num_jobs, num_machines  # These are determined by the dataset

        if self.instance_idx >= self.num_instances:
            raise IndexError("All instances from the dataset have been served.")

        # Extract the data for the current instance
        ops_durations = self.dataset[self.instance_idx, 0, :, :].astype(jnp.float32)
        #in the original dataset, the machine ids are 1-indexed
        ops_machine_ids = self.dataset[self.instance_idx, 1, :, :].astype(jnp.int32) - 1
        #NORMALIZE TO 6 the OP_DURATION(model trained with this)
        ops_durations = ops_durations / self.max_op_duration * 6
        # The number of operations for each job is constant (equal to num_machines)
        num_ops_per_job = jnp.sum(ops_machine_ids != -1, axis=1, dtype=jnp.int32)
        
        # Use a deterministic key based on the instance index for reproducibility
        instance_key = jax.random.PRNGKey(self.instance_idx)

        scenario = Scenario(
            num_jobs=self.max_num_jobs,
            num_machines=self.max_num_ops,
            max_num_jobs=self.max_num_jobs,
            max_num_ops=self.max_num_ops,
            ops_machine_ids=ops_machine_ids,
            ops_durations=ops_durations,
            num_ops_per_job=num_ops_per_job,
            key=instance_key,
        )
        
        # Move to the next instance for the subsequent call
        self.instance_idx += 1

        return scenario

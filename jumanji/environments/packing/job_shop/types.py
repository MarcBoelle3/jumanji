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

from typing import TYPE_CHECKING

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Scenario:
    """A scenario containing the description of the job shop scheduling problem.
    It is common to the constructive and improvement methods.

    num_jobs: number of jobs in the problem.
    num_machines: number of machines in the problem.
    max_num_jobs: maximum number of jobs in the problem.
    max_num_ops: maximum number of operations in the problem.
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    num_ops_per_job: for each job, it specifies the number of operations.
        Note that a 0 corresponds to a non-existing job.
    """

    num_jobs: int
    num_machines: int
    max_num_jobs: int
    max_num_ops: int
    ops_machine_ids: chex.Array  # (max_num_jobs, max_num_ops)
    ops_durations: chex.Array  # (max_num_jobs, max_num_ops)
    num_ops_per_job: chex.Array  # (max_num_jobs,)
    key: chex.PRNGKey  # (2,)


@dataclass
class CommonState:
    """The environment state containing a complete description of the job shop scheduling problem.
    It is common to the constructive and improvement methods.
    States of each method are derived from this class.

    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    step_count: used to track time, which is necessary for updating scheduled_times.
    scheduled_times: for each job, it specifies the time at which each operation was scheduled.
        Note that -1 means the operation has not been scheduled yet.
    key: random key used for auto-reset.
    """

    ops_machine_ids: chex.Array  # (max_num_jobs, max_num_ops)
    ops_durations: chex.Array  # (max_num_jobs, max_num_ops)
    step_count: chex.Numeric  # ()
    scheduled_times: chex.Array  # (max_num_jobs, max_num_ops)
    key: chex.PRNGKey  # (2,)

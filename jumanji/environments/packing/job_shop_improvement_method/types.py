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

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


class Observation(NamedTuple):
    """
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    """

    ops_machine_ids: chex.Array  # (max_num_jobs, max_num_ops)
    ops_durations: chex.Array  # (max_num_jobs, max_num_ops)
    adj_mat_pc: (
        chex.Array
    )  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2) #for source and target nodes
    adj_mat_mc: (
        chex.Array
    )  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2) #for source and target nodes
    makespan: chex.Numeric  # ()


@dataclass
class State:
    """The environment state containing a complete description of the job shop scheduling problem.
    A state contains a valid instance of the job shop scheduling problem.

    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    step_count: used to track time, which is necessary to check termination condition.
    scheduled_times: for each job, it specifies the time at which each operation was scheduled.
    adj_mat_pc: adjacency matrix of the precedence constraints graph.
    adj_mat_mc: adjacency matrix of the machine constraints graph. Updated at each step.
    makespan: the current makespan of the state.
    key: random key used for auto-reset.
    """

    ops_machine_ids: chex.Array  # (max_num_jobs, max_num_ops)
    ops_durations: chex.Array  # (max_num_jobs, max_num_ops)
    num_ops_per_job: chex.Array  # (max_num_jobs,)
    step_count: chex.Numeric  # ()
    scheduled_times: chex.Array  # (max_num_jobs, max_num_ops)
    adj_mat_pc: (
        chex.Array
    )  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2) #for source and target nodes
    adj_mat_mc: (
        chex.Array
    )  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2) #for source and target nodes
    makespan: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


@dataclass
class Scenario:
    """A scenario containing the description of the job shop scheduling problem.

    num_jobs: number of jobs in the problem.
    num_machines: number of machines in the problem.
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    num_ops_per_job: for each job, it specifies the number of operations.
        Note that a 0 corresponds to a non-existing job.
    """

    num_jobs: int
    num_machines: int
    ops_machine_ids: chex.Array  # (max_num_jobs, max_num_ops)
    ops_durations: chex.Array  # (max_num_jobs, max_num_ops)
    num_ops_per_job: chex.Array  # (max_num_jobs,)

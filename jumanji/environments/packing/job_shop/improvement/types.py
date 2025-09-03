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

from jumanji.environments.packing.job_shop.types import JobShopState

@dataclass
class BestSolution:
    """Container for storing the best solution found so far."""
    scheduled_times: chex.Array  # (max_num_jobs, max_num_ops)
    adj_mat_pc: chex.Array  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
    adj_mat_mc: chex.Array  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
    is_on_critical_path: chex.Array  # (max_num_jobs, max_num_ops)
    action_mask: chex.Array  # (max_num_ops*max_num_jobs, 2)
    operation_pairs_mask: chex.Array  # (max_num_ops*max_num_jobs, max_num_ops*max_num_jobs)
    critical_block_info: chex.Array  # (max_num_jobs * max_num_ops, 8)
    gap_left_right: chex.Array  # (max_num_jobs * max_num_ops, 2)
    est: chex.Array  # (max_num_jobs * max_num_ops + 2,)
    lst: chex.Array  # (max_num_jobs * max_num_ops + 2,)


class Observation(NamedTuple):
    """
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    adj_mat_pc: adjacency matrix of the precedence constraints graph.
    adj_mat_mc: adjacency matrix of the machine constraints graph.
    makespan: the current makespan of the state.
    """

    ops_machine_ids: chex.Array  # (num_jobs, max_num_ops)
    ops_durations: chex.Array  # (num_jobs, max_num_ops)
    edges_pc: chex.Array  # (max_num_edges, 2)
    edges_mc: chex.Array  # (max_num_edges, 2)
    makespan: chex.Numeric  # ()
    incumbent_makespan: chex.Numeric  # ()
    action_mask: chex.Array  # (max_num_ops*max_num_jobs, 2)
    observation_features: chex.Array  # (max_num_jobs * max_num_ops, 3)
    operation_pairs_mask: chex.Array  # (max_num_ops*max_num_jobs, max_num_ops*max_num_jobs)
    num_machines: chex.Array  # ()
    extra_features: chex.Array  # (max_num_ops*max_num_jobs)
    best_solution_so_far: BestSolution  # Best solution found so far  

    @property
    def agent_view(self) -> dict:
        return {
            "observation_features": self.observation_features,
            "edges_pc": self.edges_pc,
            "edges_mc": self.edges_mc,
            "action_mask": self.action_mask,
            "operation_pairs_mask": self.operation_pairs_mask,
            "makespan": self.makespan,
            "incumbent_makespan": self.incumbent_makespan,
            "num_machines": self.num_machines,
            "extra_features": self.extra_features,
            "best_solution_so_far": self.best_solution_so_far,
        }
    

@dataclass
class ImprovementState(JobShopState):
    """The environment state containing a valid schedule for a given scenario.

    num_ops_per_job: for each job, it specifies the number of operations.
        Note that a 0 corresponds to a non-existing job.
    adj_mat_pc: adjacency matrix of the precedence constraints graph.
    adj_mat_mc: adjacency matrix of the machine constraints graph. Updated at each step.
    makespan: the current makespan of the state.
    incumbent_makespan: the smallest makespan found so far in the episode.
    is_on_critical_path: for each job, it specifies whether each operation is on the critical path.
    """

    num_ops_per_job: chex.Array  # (max_num_jobs,)
    adj_mat_pc: (
        chex.Array
    )  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2) #for source and target nodes
    adj_mat_mc: (
        chex.Array
    )  # (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2) #for source and target nodes
    makespan: chex.Numeric  # ()
    incumbent_makespan: chex.Numeric  # ()
    step_minimum: chex.Numeric  # ()
    is_on_critical_path: chex.Array  # (max_num_jobs, max_num_ops)
    action_mask: chex.Array  # (max_num_ops*max_num_jobs, 2)
    critical_block_info: chex.Array  # (max_num_jobs * max_num_ops, 8)
    gap_left_right: chex.Array  # (max_num_jobs * max_num_ops, 2)
    est: chex.Array  # (max_num_jobs * max_num_ops,)
    lst: chex.Array  # (max_num_jobs * max_num_ops,)
    operation_pairs_mask: chex.Array  # (max_num_ops*max_num_jobs, max_num_ops*max_num_jobs)
    best_solution_so_far: BestSolution  # Best solution found so far
    step_since_best: chex.Numeric  # Steps since last improvement
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
from typing import Optional

import chex


class AbstractSchedulingMethod(abc.ABC):
    """Base class for job shop scheduling methods."""

    # Class-level name attribute
    NAME: str = NotImplemented

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        max_num_jobs: int,
        max_num_ops: int,
    ):
        """Initialize scheduling method with problem dimensions.

        Args:
            num_jobs: Number of jobs to schedule.
            num_machines: Number of machines available.
            max_num_jobs: Maximum number of jobs (static parameter).
            max_num_ops: Maximum number of operations per job (static parameter).
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_num_jobs = max_num_jobs
        self.max_num_ops = max_num_ops

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the scheduling method."""

    @abc.abstractmethod
    def build_machine_adjacency_matrix(
        self,
        ops_machine_ids: chex.Array,
        ops_durations: chex.Array,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """Build machine constraint adjacency matrix.

        Creates an adjacency matrix encoding machine precedence constraints
        based on the scheduling method's operation ordering decisions.

        Args:
            ops_machine_ids: Machine assignments (max_num_jobs, max_num_ops),
                            -1 for padding.
            ops_durations: Operation durations (max_num_jobs, max_num_ops),
                          -1 for padding.
            key: Random key for stochastic methods.

        Returns:
            Machine constraint adjacency matrix
            (max_num_jobs*max_num_ops+2, max_num_jobs*max_num_ops+2)
            with source and target nodes included.
        """

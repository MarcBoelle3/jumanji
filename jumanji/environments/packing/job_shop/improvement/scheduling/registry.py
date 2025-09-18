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

from collections import OrderedDict
from typing import List, Type

from jumanji.environments.packing.job_shop.improvement.scheduling.base import (
    AbstractSchedulingMethod,
)

# Store method classes in order of registration
_REGISTRY = OrderedDict()


class MethodRegistry:
    """Registry for scheduling methods with automatic discovery and ordering."""

    @staticmethod
    def register(cls: Type[AbstractSchedulingMethod]) -> Type[AbstractSchedulingMethod]:
        """Register a scheduling method class.

        Args:
            cls: The scheduling method class to register.

        Returns:
            The method class (for use as a decorator).

        Raises:
            ValueError: If method name is already registered.
        """
        # Create temporary instance to get name
        temp_instance = cls(num_jobs=1, num_machines=1, max_num_jobs=1, max_num_ops=1)
        method_name = temp_instance.name

        if method_name in _REGISTRY:
            raise ValueError(f"Method {method_name} already registered.")
        _REGISTRY[method_name] = cls
        return cls

    @staticmethod
    def get_method_names() -> List[str]:
        """Returns an ordered list of registered method names."""
        return list(_REGISTRY.keys())

    @staticmethod
    def get_instances(
        num_jobs: int, num_machines: int, max_num_jobs: int, max_num_ops: int
    ) -> List[AbstractSchedulingMethod]:
        """Returns an ordered list of registered method instances with given parameters.

        Args:
            num_jobs: Number of jobs to schedule.
            num_machines: Number of machines available.
            max_num_jobs: Maximum number of jobs (static parameter).
            max_num_ops: Maximum number of operations per job (static parameter).

        Returns:
            List of instantiated scheduling methods in registration order.
        """
        return [
            cls(num_jobs, num_machines, max_num_jobs, max_num_ops) for cls in _REGISTRY.values()
        ]

    @staticmethod
    def get_method_index(name: str) -> int:
        """Returns the integer index for a given method name.

        Args:
            name: The name of the scheduling method.

        Returns:
            Integer index of the method in registration order.

        Raises:
            ValueError: If method name is not found.
        """
        try:
            return list(_REGISTRY.keys()).index(name)
        except ValueError:
            available_methods = list(_REGISTRY.keys())
            raise ValueError(
                f"Method '{name}' not found in registry. Available: {available_methods}"
            ) from None

    @staticmethod
    def get_method(
        name: str, num_jobs: int, num_machines: int, max_num_jobs: int, max_num_ops: int
    ) -> AbstractSchedulingMethod:
        """Get a scheduling method instance by name.

        Args:
            name: The name of the scheduling method.
            num_jobs: Number of jobs to schedule.
            num_machines: Number of machines available.
            max_num_jobs: Maximum number of jobs (static parameter).
            max_num_ops: Maximum number of operations per job (static parameter).

        Returns:
            The scheduling method instance.

        Raises:
            ValueError: If the method name is not registered.
        """
        if name not in _REGISTRY:
            available_methods = list(_REGISTRY.keys())
            raise ValueError(f"Method '{name}' not found. Available methods: {available_methods}")
        cls = _REGISTRY[name]
        return cls(num_jobs, num_machines, max_num_jobs, max_num_ops)

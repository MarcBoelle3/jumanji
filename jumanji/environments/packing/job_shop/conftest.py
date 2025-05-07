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

import pytest

from jumanji.environments.packing.job_shop.scenario_generator import RandomScenarioGenerator


@pytest.fixture
def dummy_scenario_generator() -> RandomScenarioGenerator:
    """Create a dummy RandomScenarioGenerator with fixed parameters."""
    max_num_jobs = 5
    max_num_ops = 3
    max_op_duration = 4
    return RandomScenarioGenerator(max_num_jobs, max_num_ops, max_op_duration)

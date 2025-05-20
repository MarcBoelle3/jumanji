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

from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jumanji.environments.packing.job_shop.improvement.compute_makespan import (
    forward_backward_pass_jraph,
)
from jumanji.environments.packing.job_shop.improvement.env import JobShop
from jumanji.environments.packing.job_shop.improvement.generator import RandomScheduleGenerator
from jumanji.environments.packing.job_shop.improvement.get_actions import (
    choose_random_action_N5,
    get_critical_operations,
)
from jumanji.environments.packing.job_shop.scenario_generator import RandomScenarioGenerator
from jumanji.environments.packing.job_shop.viewer import JobShopViewer


def test_jobshop_viewer(job_shop_env: JobShop) -> None:
    """Test that the JobShop viewer can render states correctly."""
    # Get initial state
    key = jax.random.PRNGKey(0)
    state = job_shop_env.reset(key)

    # Create viewer
    viewer = JobShopViewer(
        name="JobShop",
        num_jobs=job_shop_env.scenario_generator.max_num_jobs,
        num_machines=job_shop_env.scenario_generator.num_machines,
        max_num_ops=job_shop_env.scenario_generator.max_num_ops,
        max_op_duration=job_shop_env.scenario_generator.max_op_duration,
        render_mode="human",
    )

    # Create two states at different timesteps
    action = jnp.array([6, 5], jnp.int32)
    state1, next_timestep = job_shop_env.step(state, action)

    next_action = jnp.array([2, 7], jnp.int32)
    state2, next_timestep = job_shop_env.step(state=state1, action=next_action)

    # Test animation with two frames
    animation = viewer.animate(
        [state, state1, state2], interval=500, save_path="test_animation.gif"
    )
    plt.show()


def test_jobshop_viewer_random_actions() -> None:
    """Test that the JobShop viewer can render states with random actions."""
    # Initialize dummy generator and environment
    scenario_generator = RandomScenarioGenerator(max_num_jobs=25, max_num_ops=8, max_op_duration=8)

    schedule_generator = RandomScheduleGenerator(
        num_jobs=20, num_machines=10, max_num_jobs=25, max_num_ops=8
    )
    job_shop_env = JobShop(scenario_generator, schedule_generator)

    # Get initial state and key
    key = jax.random.PRNGKey(42)
    state, _ = job_shop_env.reset(key)

    # Create viewer
    viewer = JobShopViewer(
        name="JobShop Random Actions",
        num_jobs=job_shop_env.scenario_generator.max_num_jobs,
        num_machines=job_shop_env.schedule_generator.num_machines,
        max_num_ops=job_shop_env.scenario_generator.max_num_ops,
        max_op_duration=job_shop_env.scenario_generator.max_op_duration,
        render_mode="human",
        show_critical_path=True,
    )

    # Store states for animation
    states = [state]
    # Record execution time for each step
    start_time = time()

    # JIT compile the step function and forward_backward_pass for better performance
    jitted_step = jax.jit(job_shop_env.step)
    jitted_forward_backward_pass = jax.jit(
        lambda adj_mat, ops_durations: forward_backward_pass_jraph(
            adj_mat, ops_durations, job_shop_env.max_num_edges
        )
    )

    # Take 10 random steps
    for _ in range(30):
        # Get critical operations and choose random action
        adj_mat = jnp.maximum(state.adj_mat_pc, state.adj_mat_mc)
        est, lst, _ = jitted_forward_backward_pass(adj_mat, state.ops_durations)
        jax.debug.print("forward_backward_pass complete")
        critical_ops = get_critical_operations(
            est,
            lst,
            state.adj_mat_mc,
            state.ops_durations,
            job_shop_env.scenario_generator.max_num_jobs,
            job_shop_env.scenario_generator.max_num_ops,
            job_shop_env.max_num_edges,
        )
        jax.debug.print("get_critical_operations complete")
        key, subkey = jax.random.split(key)
        action = choose_random_action_N5(
            subkey, critical_ops, job_shop_env.scenario_generator.max_num_ops
        )
        jax.debug.print("action: {}", action)
        jax.debug.print("choose_random_action complete")
        # Take step and store new state
        state, _ = jitted_step(state, action)
        jax.debug.print("step complete")
        states.append(state)
        jax.debug.print("adj mat mc: {}", state.adj_mat_mc)
    jax.debug.breakpoint()

    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")
    # Create and show animation
    animation = viewer.animate(
        states, interval=500, save_path="random_actions_animation_N5_empty.gif"
    )
    plt.show()


if __name__ == "__main__":
    test_jobshop_viewer_random_actions()

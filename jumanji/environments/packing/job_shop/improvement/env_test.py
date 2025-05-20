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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.packing.job_shop.improvement.env import JobShop
from jumanji.environments.packing.job_shop.improvement.types import ImprovementState
from jumanji.environments.packing.job_shop.improvement.update_sol import update_disjunctive_graph
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
    make_masked_categorical_random_ndim_for_test,
)
from jumanji.types import TimeStep


class TestJobShop:
    def test_job_shop__reset(self, job_shop_env: JobShop) -> None:
        """Test that the environment is reset correctly."""
        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        assert jnp.all(
            state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )
        assert jnp.all(
            state.scheduled_times
            == jnp.array(
                [
                    [0, 3, 6],
                    [3, 5, 9],
                    [5, 9, -1],
                ],
                jnp.int32,
            )
        )
        assert jnp.all(state.num_ops_per_job == jnp.array([3, 3, 2], jnp.int32))
        assert state.step_count == jnp.array(0, jnp.int32)

    def test_job_shop__reset_jit(self, job_shop_env: JobShop) -> None:
        """Confirm that the reset is only compiled once when jitted."""
        chex.clear_trace_counter()
        reset_fn = jax.jit(chex.assert_max_traces(job_shop_env.reset, n=1))
        key = jax.random.PRNGKey(0)
        state, timestep = reset_fn(key)

        # Call again to check it does not compile twice
        state, timestep = reset_fn(key)
        assert isinstance(timestep, TimeStep)
        assert isinstance(state, ImprovementState)

    def test_job_shop_adj_mat_pc(self, job_shop_env: JobShop) -> None:
        key = jax.random.PRNGKey(0)
        state, _ = job_shop_env.reset(key)
        jax.debug.print("state.adj_mat_pc: {}", (state.adj_mat_pc,))
        assert jnp.all(
            state.adj_mat_pc
            == jnp.array(
                [
                    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                    [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

    def test_job_shop_update_disjunctive_graph(self, job_shop_env: JobShop) -> None:
        key = jax.random.PRNGKey(0)
        state, _ = job_shop_env.reset(key)

        assert jnp.all(
            state.adj_mat_mc
            == jnp.array(
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
        )

        # Test that the update is the same for both actions
        dummy_ops_pair1 = jnp.array(
            [6, 5, 0], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 5 coming left to 6
        dummy_ops_pair2 = jnp.array(
            [6, 5, 1], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 6 coming right to 5
        new_adj_mat_mc = update_disjunctive_graph(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair1
        )
        new_adj_mat_mc_2 = update_disjunctive_graph(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair2
        )
        assert jnp.all(
            new_adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

        assert jnp.all(new_adj_mat_mc_2 == new_adj_mat_mc)

    def test_job_shop_update_disjunctive_graph_jitted(self, job_shop_env: JobShop) -> None:
        key = jax.random.PRNGKey(0)
        state, _ = job_shop_env.reset(key)

        dummy_ops_pair1 = jnp.array(
            [6, 5, 0], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 5 coming left to 6
        dummy_ops_pair2 = jnp.array(
            [6, 5, 1], jnp.int32
        )  # change 6 -> 5 into 5 -> 6, with 6 coming right to 5
        new_adj_mat_mc = jax.jit(update_disjunctive_graph)(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair1
        )
        new_adj_mat_mc_2 = jax.jit(update_disjunctive_graph)(
            state.adj_mat_mc, state.ops_durations, dummy_ops_pair2
        )
        assert jnp.all(
            new_adj_mat_mc
            == jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )
        assert jnp.all(new_adj_mat_mc_2 == new_adj_mat_mc)

    def test_job_shop__step(self, job_shop_env: JobShop) -> None:
        """Test 2 steps of the dummy instance."""

        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        # STEP T=0 -> T=1
        # Action: switch operation 6 to the right
        action = jnp.array([6, 1], jnp.int32)
        next_state, next_timestep = job_shop_env.step(state, action)

        assert jnp.all(
            next_state.ops_machine_ids
            == jnp.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 2, -1],
                ]
            )
        )
        assert jnp.all(
            next_state.ops_durations
            == jnp.array(
                [
                    [3, 2, 2],
                    [2, 1, 4],
                    [4, 3, -1],
                ]
            )
        )

        assert jnp.all(
            next_state.scheduled_times
            == jnp.array(
                [
                    [0, 3, 6],
                    [3, 5, 6],
                    [10, 14, -1],
                ],
                jnp.int32,
            )
        )
        assert next_state.step_count == 1
        assert next_state.makespan == 17

    def test_job_shop__step_jit(self, job_shop_env: JobShop) -> None:
        """Confirm that the step is only compiled once when jitted."""
        key = jax.random.PRNGKey(0)
        state, timestep = job_shop_env.reset(key)

        # Action : switch operation 6 to the right
        action = jnp.array([6, 1], jnp.int32)
        jax.debug.print("action: {}", (action[0], action[1]))
        chex.clear_trace_counter()
        step_fn = jax.jit(chex.assert_max_traces(job_shop_env.step, n=1))
        _, _ = step_fn(state, action)

        # Call again to check it does not compile twice
        next_state, next_timestep = step_fn(state, action)
        assert isinstance(next_timestep, TimeStep)
        assert isinstance(next_state, ImprovementState)

    def test_job_shop_env__does_not_smoke(self, job_shop_env: JobShop) -> None:
        """Test that we can run an episode without any errors."""
        check_env_does_not_smoke(
            job_shop_env,
            select_action=make_masked_categorical_random_ndim_for_test(
                job_shop_env.action_spec.num_values
            ),
        )

    def test_job_shop_env__specs_does_not_smoke(self, job_shop_env: JobShop) -> None:
        """Test that we can access specs without any errors."""
        check_env_specs_does_not_smoke(job_shop_env)

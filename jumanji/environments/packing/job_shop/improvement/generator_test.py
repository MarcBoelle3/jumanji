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
import pytest

from jumanji.environments.packing.job_shop.conftest import DummyScenarioGenerator
from jumanji.environments.packing.job_shop.improvement.conftest import DummyScheduleGenerator
from jumanji.environments.packing.job_shop.improvement.generator import (
    ScheduleGenerator,
)
from jumanji.environments.packing.job_shop.improvement.scheduling import MethodRegistry
from jumanji.environments.packing.job_shop.improvement.types import (
    ImprovementState,
    Neighborhood,
    SchedulingMethod,
)
from jumanji.environments.packing.job_shop.types import Scenario
from jumanji.testing.pytrees import assert_trees_are_equal


class TestMatrixGenerator:
    """Test basic matrix generation functionality using DummyScheduleGenerator."""

    def test_init_adj_mat_pc(
        self,
        dummy_scenario_generator: DummyScenarioGenerator,
        dummy_schedule_generator: DummyScheduleGenerator,
    ) -> None:
        """Test that the precedence constraint matrix is correctly initialized."""
        key = jax.random.PRNGKey(0)
        scenario = dummy_scenario_generator(key, num_jobs=3, num_machines=3)

        adj_mat_pc = dummy_schedule_generator.build_precedence_matrix(
            scenario.ops_durations, scenario.num_ops_per_job
        )

        assert adj_mat_pc.shape == (
            scenario.max_num_jobs * scenario.max_num_ops + 2,
            scenario.max_num_jobs * scenario.max_num_ops + 2,
        )

        # Verify precedence constraints: source (0) -> first operation of each job
        for job_id in range(3):
            first_op = 1 + job_id * scenario.max_num_ops
            assert adj_mat_pc[0, first_op] > 0  # Source to first operation

        # Verify job flow: operation -> next operation in same job
        for job_id in range(3):
            num_ops = scenario.num_ops_per_job[job_id]
            for op_idx in range(num_ops - 1):
                curr_op = 1 + job_id * scenario.max_num_ops + op_idx
                next_op = curr_op + 1
                assert adj_mat_pc[curr_op, next_op] > 0

        # Count total links: should equal total number of operations
        num_total_ops_links = jnp.sum(adj_mat_pc[1:, 1:] > 0)
        assert num_total_ops_links == jnp.sum(scenario.num_ops_per_job)

    def test_init_adj_mat_pc__jit(
        self,
        dummy_schedule_generator: DummyScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that precedence matrix initialization is JIT-compatible."""
        scenario = dummy_scenario_generator(jax.random.PRNGKey(0), num_jobs=3, num_machines=3)

        @jax.jit
        def build_precedence_jit(
            ops_durations: chex.Array, num_ops_per_job: chex.Array
        ) -> chex.Array:
            return dummy_schedule_generator.build_precedence_matrix(ops_durations, num_ops_per_job)

        adj_mat_pc_jit = build_precedence_jit(scenario.ops_durations, scenario.num_ops_per_job)
        adj_mat_pc_normal = dummy_schedule_generator.build_precedence_matrix(
            scenario.ops_durations, scenario.num_ops_per_job
        )

        assert jnp.allclose(adj_mat_pc_jit, adj_mat_pc_normal)

    def test_dummy_generator_deterministic(
        self,
        dummy_schedule_generator: DummyScheduleGenerator,
        dummy_scenario_generator: DummyScenarioGenerator,
    ) -> None:
        """Test that DummyScheduleGenerator produces deterministic results."""
        scenario = dummy_scenario_generator(jax.random.PRNGKey(0), num_jobs=3, num_machines=3)

        chex.clear_trace_counter()
        # Test that call method gives same results for same keys
        state1 = dummy_schedule_generator(
            jax.random.PRNGKey(1), scenario, SchedulingMethod.PRIORITY_LIST, Neighborhood.N5
        )
        state2 = dummy_schedule_generator(
            jax.random.PRNGKey(2), scenario, SchedulingMethod.PRIORITY_LIST, Neighborhood.N5
        )
        assert_trees_are_equal(state1, state2)


class TestDisjunctiveGraph:
    """Test properties of the disjunctive graph (max of precedence and machine constraints)."""

    @pytest.fixture
    def schedule_generator(self) -> ScheduleGenerator:
        return ScheduleGenerator(num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3)

    @pytest.fixture
    def test_scenario(self, dummy_scenario_generator: DummyScenarioGenerator) -> Scenario:
        key = jax.random.PRNGKey(42)
        return dummy_scenario_generator(key, num_jobs=3, num_machines=3)

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_disjunctive_graph_no_self_loops(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that the disjunctive graph has no self-loops."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Create disjunctive graph as max of precedence and machine constraints
        disjunctive_graph = jnp.maximum(state.adj_mat_pc, state.adj_mat_mc)

        # No self-loops (diagonal should be zero)
        assert jnp.all(
            jnp.diag(disjunctive_graph) == 0
        ), "Disjunctive graph should have no self-loops"

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_disjunctive_graph_no_cycles(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that the disjunctive graph has no cycles."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Create disjunctive graph as max of precedence and machine constraints
        disjunctive_graph = jnp.maximum(state.adj_mat_pc, state.adj_mat_mc)

        # Test for cycles using matrix powers - if A^n has non-zero diagonal, there are cycles
        n = disjunctive_graph.shape[0]
        current_power = disjunctive_graph
        for _ in range(n):
            # Check diagonal of current power
            diagonal = jnp.diag(current_power)
            # Only check non-zero entries (where cycles could exist)
            cycle_detected = jnp.any(diagonal > 0)
            assert not cycle_detected, f"Cycle detected in disjunctive graph for {method.name}"
            # Compute next power for next iteration
            current_power = current_power @ disjunctive_graph

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_disjunctive_graph_node_connectivity(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test node connectivity constraints in the disjunctive graph."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Create disjunctive graph as max of precedence and machine constraints
        disjunctive_graph = jnp.maximum(state.adj_mat_pc, state.adj_mat_mc)

        # Convert to binary matrix (>0 means edge exists)
        binary_graph = disjunctive_graph > 0

        # Get edge counts
        outgoing_edges = jnp.sum(binary_graph, axis=1)
        incoming_edges = jnp.sum(binary_graph, axis=0)

        num_nodes = binary_graph.shape[0]
        source_idx = 0
        target_idx = num_nodes - 1

        # Test source node: should have exactly num_jobs outgoing edges
        # (one to each job's first operation)
        source_outgoing = outgoing_edges[source_idx]
        assert (
            source_outgoing == test_scenario.num_jobs
        ), f"Source should have {test_scenario.num_jobs} outgoing edges, got {source_outgoing}"

        # Source should have no incoming edges
        source_incoming = incoming_edges[source_idx]
        assert source_incoming == 0, f"Source should have 0 incoming edges, got {source_incoming}"

        # Test target node: should have exactly num_jobs incoming edges
        # (one from each job's last operation)
        target_incoming = incoming_edges[target_idx]
        assert (
            target_incoming == test_scenario.num_jobs
        ), f"Target should have {test_scenario.num_jobs} incoming edges, got {target_incoming}"

        # Target should have no outgoing edges
        target_outgoing = outgoing_edges[target_idx]
        assert target_outgoing == 0, f"Target should have 0 outgoing edges, got {target_outgoing}"

        # Test operation nodes (excluding source and target):
        # at most 2 incoming and 2 outgoing edges
        for node_idx in range(1, num_nodes - 1):
            node_outgoing = outgoing_edges[node_idx]
            node_incoming = incoming_edges[node_idx]

            assert (
                node_outgoing <= 2
            ), f"Operation node {node_idx} has more than 2 outgoing edges: {node_outgoing}"
            assert (
                node_incoming <= 2
            ), f"Operation node {node_idx} has more than 2 incoming edges: {node_incoming}"

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_disjunctive_graph_invalid_operations_isolation(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that invalid operations are isolated in the disjunctive graph."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Create disjunctive graph as max of precedence and machine constraints
        disjunctive_graph = jnp.maximum(state.adj_mat_pc, state.adj_mat_mc)

        # Find invalid operations (those with machine_id = -1)
        max_num_jobs, max_num_ops = test_scenario.ops_machine_ids.shape

        for job_id in range(max_num_jobs):
            for op_id in range(max_num_ops):
                if int(test_scenario.ops_machine_ids[job_id, op_id]) == -1:
                    # This is an invalid operation
                    node_idx = job_id * max_num_ops + op_id + 1  # +1 for source node offset

                    # Invalid operation should have no outgoing edges
                    outgoing_edges = jnp.sum(disjunctive_graph[node_idx, :])
                    assert (
                        outgoing_edges == 0
                    ), f"Invalid operation at job {job_id}, op {op_id} has outgoing edges"

                    # Invalid operation should have no incoming edges (except possibly from source)
                    incoming_edges = jnp.sum(disjunctive_graph[:, node_idx])
                    if incoming_edges > 0:
                        # Only source should connect to invalid operations,
                        # and only if it's the first op
                        source_connection = disjunctive_graph[0, node_idx]
                        assert source_connection > 0 and op_id == 0, (
                            f"Invalid operation at job {job_id}, op {op_id} "
                            f"has non-source incoming edges"
                        )


class TestScheduleGenerator:
    """Comprehensive tests for the ScheduleGenerator."""

    @pytest.fixture
    def schedule_generator(self) -> ScheduleGenerator:
        """Create a ScheduleGenerator with automatic method discovery."""
        return ScheduleGenerator(
            num_jobs=3,
            num_machines=3,
            max_num_jobs=3,
            max_num_ops=3,
        )

    @pytest.fixture
    def test_scenario(self, dummy_scenario_generator: DummyScenarioGenerator) -> Scenario:
        """Create a test scenario for consistent testing."""
        key = jax.random.PRNGKey(42)
        return dummy_scenario_generator(key, num_jobs=3, num_machines=3)

    def test_simplified_api_instantiation(self, schedule_generator: ScheduleGenerator) -> None:
        """Test that the simplified API works correctly."""
        # Should automatically discover and instantiate all registered methods
        assert len(schedule_generator.scheduling_methods) == len(MethodRegistry.get_method_names())

        # Verify properties
        assert schedule_generator.num_jobs == 3
        assert schedule_generator.num_machines == 3
        assert schedule_generator.max_num_jobs == 3
        assert schedule_generator.max_num_ops == 3

        # Verify that all methods are properly initialized with correct parameters
        for method in schedule_generator.scheduling_methods:
            assert method.num_jobs == 3
            assert method.num_machines == 3
            assert method.max_num_jobs == 3
            assert method.max_num_ops == 3

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_method_switching(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that the generator can switch between different scheduling methods."""
        key = jax.random.PRNGKey(42)

        # Generate state with the specified method
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Verify the state is valid
        assert isinstance(state, ImprovementState)
        assert state.ops_machine_ids.shape == (3, 3)
        assert state.ops_durations.shape == (3, 3)
        assert state.num_ops_per_job.shape == (3,)
        assert state.step_count == 0

        # Verify adjacency matrices have correct shape
        expected_size = 3 * 3 + 2  # max_num_jobs * max_num_ops + 2
        assert state.adj_mat_pc.shape == (expected_size, expected_size)
        assert state.adj_mat_mc.shape == (expected_size, expected_size)

    @pytest.mark.parametrize(
        "method",
        [
            SchedulingMethod.SHORTEST_PROCESSING_TIME,
            SchedulingMethod.FLOW_DUE_DATE_MOST_WORK,
        ],
    )
    def test_deterministic_methods_consistency(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that deterministic methods produce consistent results."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(84)

        state1 = schedule_generator(key1, test_scenario, method, Neighborhood.N5)
        state2 = schedule_generator(key2, test_scenario, method, Neighborhood.N5)

        # Deterministic methods should produce identical results regardless of key
        assert jnp.allclose(state1.adj_mat_pc, state2.adj_mat_pc)
        assert jnp.allclose(state1.adj_mat_mc, state2.adj_mat_mc)
        assert jnp.allclose(state1.scheduled_times, state2.scheduled_times)

    def test_best_solution_initialization(
        self, schedule_generator: ScheduleGenerator, test_scenario: Scenario
    ) -> None:
        """Test that best_solution_so_far is correctly initialized."""
        key = jax.random.PRNGKey(42)

        # Test all methods from registry
        for method in SchedulingMethod:
            state = schedule_generator(key, test_scenario, method, Neighborhood.N5)
            best = state.best_solution_so_far

            # Best solution should mirror the initially generated state
            assert jnp.allclose(best.scheduled_times, state.scheduled_times)
            assert jnp.allclose(best.adj_mat_pc, state.adj_mat_pc)
            assert jnp.allclose(best.adj_mat_mc, state.adj_mat_mc)
            assert jnp.allclose(best.action_mask, state.action_mask)
            assert jnp.allclose(best.critical_block_info, state.critical_block_info)
            assert jnp.allclose(best.gap_left_right, state.gap_left_right)
            assert jnp.allclose(best.est, state.est)
            assert jnp.allclose(best.lst, state.lst)


class TestJitCompatibility:
    """Test JAX JIT compatibility of the ScheduleGenerator."""

    @pytest.fixture
    def schedule_generator(self) -> ScheduleGenerator:
        return ScheduleGenerator(
            num_jobs=3,
            num_machines=3,
            max_num_jobs=3,
            max_num_ops=3,
        )

    @pytest.fixture
    def test_scenario(self, dummy_scenario_generator: DummyScenarioGenerator) -> Scenario:
        key = jax.random.PRNGKey(42)
        return dummy_scenario_generator(key, num_jobs=3, num_machines=3)

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_generator_jit_compilation(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that the generator can be JIT-compiled and only compiles once."""

        # Create a wrapper function that can be traced
        def generate_fn(
            key: chex.PRNGKey,
            scenario: Scenario,
            method_enum: SchedulingMethod,
            neighborhood: Neighborhood,
        ) -> ImprovementState:
            return schedule_generator(key, scenario, method_enum, neighborhood)

        # Clear trace counter and create JIT-compiled function with max traces assertion
        chex.clear_trace_counter()
        jit_fn = jax.jit(chex.assert_max_traces(generate_fn, n=1))

        # First call should trigger compilation
        state1 = jit_fn(jax.random.PRNGKey(1), test_scenario, method, Neighborhood.N5)

        # Second call should use cached compilation
        state2 = jit_fn(jax.random.PRNGKey(2), test_scenario, method, Neighborhood.N5)

        # Verify results are valid
        assert isinstance(state1, ImprovementState)
        assert isinstance(state2, ImprovementState)

    def test_method_switching_under_jit(
        self, schedule_generator: ScheduleGenerator, test_scenario: Scenario
    ) -> None:
        """Test that method switching works correctly under JIT."""

        # Create a wrapper function for method switching
        def generate_with_method(method: SchedulingMethod) -> ImprovementState:
            return schedule_generator(
                jax.random.PRNGKey(42), test_scenario, method, Neighborhood.N5
            )

        # Clear trace counter and create JIT-compiled function with max traces assertion
        chex.clear_trace_counter()
        jit_fn = jax.jit(chex.assert_max_traces(generate_with_method, n=1))

        # Test all methods - should compile only once due to method switching logic
        states = {}
        for method in SchedulingMethod:
            states[method] = jit_fn(method)
            assert isinstance(states[method], ImprovementState)

        # Deterministic methods should produce same results each time
        for method in SchedulingMethod:
            if method != SchedulingMethod.PRIORITY_LIST:
                state2 = jit_fn(method)
                assert_trees_are_equal(states[method], state2)

    def test_jit_with_different_neighborhoods(
        self, schedule_generator: ScheduleGenerator, test_scenario: Scenario
    ) -> None:
        """Test JIT compilation with different neighborhood types."""

        def generate_with_neighborhood(neighborhood: Neighborhood) -> ImprovementState:
            return schedule_generator(
                jax.random.PRNGKey(42),
                test_scenario,
                SchedulingMethod.SHORTEST_PROCESSING_TIME,
                neighborhood,
            )

        chex.clear_trace_counter()
        jit_fn = jax.jit(chex.assert_max_traces(generate_with_neighborhood, n=1))

        state_n5 = jit_fn(Neighborhood.N5)
        state_n6 = jit_fn(Neighborhood.N6)

        assert isinstance(state_n5, ImprovementState)
        assert isinstance(state_n6, ImprovementState)

        # Should produce different action masks
        assert not jnp.allclose(state_n5.action_mask, state_n6.action_mask)

    def test_compilation_tracing_efficiency(
        self, schedule_generator: ScheduleGenerator, test_scenario: Scenario
    ) -> None:
        """Test that JIT compilation is efficient and doesn't retrace unnecessarily."""

        def generate_state(method: SchedulingMethod) -> ImprovementState:
            return schedule_generator(
                jax.random.PRNGKey(42), test_scenario, method, Neighborhood.N5
            )

        # Correct order: assert_max_traces first, then jax.jit
        jit_generate_state = jax.jit(chex.assert_max_traces(generate_state, n=3))

        chex.clear_trace_counter()

        # Test multiple calls - should not exceed trace limit
        for method in SchedulingMethod:
            state = jit_generate_state(method)
            assert isinstance(state, ImprovementState)

            # Second call with same method should use cached compilation
            state2 = jit_generate_state(method)
            assert isinstance(state2, ImprovementState)


class TestScheduleValidation:
    """Test that generated schedules are valid and respect constraints."""

    @pytest.fixture
    def schedule_generator(self) -> ScheduleGenerator:
        return ScheduleGenerator(num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3)

    @pytest.fixture
    def test_scenario(self, dummy_scenario_generator: DummyScenarioGenerator) -> Scenario:
        key = jax.random.PRNGKey(42)
        return dummy_scenario_generator(key, num_jobs=3, num_machines=3)

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_schedule_respects_precedence_constraints(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that generated schedules respect job precedence constraints."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # For each job, verify that operations are scheduled in order
        for job_id in range(state.ops_machine_ids.shape[0]):
            job_ops_valid = state.ops_machine_ids[job_id] != -1

            # Convert JAX array to Python bool for if condition
            if not bool(jnp.any(job_ops_valid)):
                continue  # Skip fully padded jobs

            job_start_times = []
            for op_id in range(state.ops_machine_ids.shape[1]):
                # Convert to Python bool for if condition
                if bool(job_ops_valid[op_id]):
                    # Get start time directly from scheduled_times matrix
                    start_time = state.scheduled_times[job_id, op_id]
                    job_start_times.append(float(start_time))  # Convert to Python float

            # Verify start times are in increasing order
            for i in range(len(job_start_times) - 1):
                assert (
                    job_start_times[i] <= job_start_times[i + 1]
                ), f"Job {job_id} operations not scheduled in precedence order: {job_start_times}"

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_schedule_respects_machine_constraints(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that no two operations on the same machine overlap."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Group operations by machine
        machine_operations: dict[int, list[tuple[float, float, int, int]]] = {}
        for job_id in range(state.ops_machine_ids.shape[0]):
            for op_id in range(state.ops_machine_ids.shape[1]):
                machine_id = state.ops_machine_ids[job_id, op_id]
                # Convert JAX scalar to Python int for hashable dict key
                machine_id_int = int(machine_id)
                if machine_id_int != -1:  # Valid operation
                    start_time = state.scheduled_times[job_id, op_id]
                    duration = state.ops_durations[job_id, op_id]
                    end_time = start_time + duration

                    if machine_id_int not in machine_operations:
                        machine_operations[machine_id_int] = []
                    machine_operations[machine_id_int].append(
                        (float(start_time), float(end_time), job_id, op_id)
                    )

        # Check for overlaps on each machine
        for machine_id, operations in machine_operations.items():
            # Sort by start time
            operations.sort(key=lambda x: x[0])

            for i in range(len(operations) - 1):
                current_end = operations[i][1]
                next_start = operations[i + 1][0]
                assert current_end <= next_start, (
                    f"Operations overlap on machine {machine_id}: "
                    f"Op ({operations[i][2]}, {operations[i][3]}) ends at {current_end}, "
                    f"Op ({operations[i+1][2]}, {operations[i+1][3]}) starts at {next_start}"
                )

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_makespan_calculation(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that makespan is correctly calculated."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Calculate expected makespan
        max_end_time = 0.0
        for job_id in range(state.ops_machine_ids.shape[0]):
            for op_id in range(state.ops_machine_ids.shape[1]):
                # Convert to Python int for comparison
                if int(state.ops_machine_ids[job_id, op_id]) != -1:
                    start_time = state.scheduled_times[job_id, op_id]
                    duration = state.ops_durations[job_id, op_id]
                    end_time = start_time + duration
                    max_end_time = max(max_end_time, float(end_time))

        # Use state.makespan instead of indexing scheduled_times
        calculated_makespan = float(state.makespan)

        assert (
            abs(calculated_makespan - max_end_time) < 1e-5
        ), f"Makespan mismatch: calculated={calculated_makespan}, expected={max_end_time}"

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_schedule_times_are_valid(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that all scheduled times are non-negative and finite."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Check valid operations have non-negative times
        valid_mask = state.ops_machine_ids != -1
        valid_times = state.scheduled_times[valid_mask]
        assert jnp.all(valid_times >= 0), "Some valid scheduled times are negative"

        # All valid scheduled times should be finite
        assert jnp.all(jnp.isfinite(valid_times)), "Some valid scheduled times are not finite"

        # Padded operations should have -1 times
        padded_mask = state.ops_machine_ids == -1
        if jnp.any(padded_mask):
            padded_times = state.scheduled_times[padded_mask]
            assert jnp.all(padded_times == -1), "Padded operations should have -1 scheduled time"

    @pytest.mark.parametrize("method", list(SchedulingMethod))
    def test_critical_path_analysis(
        self,
        schedule_generator: ScheduleGenerator,
        test_scenario: Scenario,
        method: SchedulingMethod,
    ) -> None:
        """Test that critical path analysis produces reasonable results."""
        key = jax.random.PRNGKey(42)
        state = schedule_generator(key, test_scenario, method, Neighborhood.N5)

        # Critical path info should have correct shape (max_num_jobs * max_num_ops)
        expected_ops = state.ops_machine_ids.shape[0] * state.ops_machine_ids.shape[1]  # No +2
        assert state.critical_block_info.shape[0] == expected_ops

        # is_on_critical_path should be boolean-like
        assert jnp.all((state.is_on_critical_path == 0) | (state.is_on_critical_path == 1))

        # Test that the sum of durations of operations on the critical path >= makespan
        critical_path_mask = state.is_on_critical_path == 1
        if jnp.any(critical_path_mask):
            # Reshape critical path mask to match ops_durations shape [max_num_jobs, max_num_ops]
            critical_path_2d = critical_path_mask.reshape(state.ops_durations.shape)
            # Get durations of operations on critical path
            critical_durations = state.ops_durations[critical_path_2d]
            # Filter out invalid operations (duration -1)
            valid_critical_durations = critical_durations[critical_durations > 0]

            if len(valid_critical_durations) > 0:
                critical_path_duration = jnp.sum(valid_critical_durations)
                # Critical path duration should be >= makespan (allowing for small numerical errors)
                assert critical_path_duration >= state.makespan - 1e-5, (
                    f"Critical path duration {critical_path_duration} "
                    f"should be >= makespan {state.makespan}"
                )

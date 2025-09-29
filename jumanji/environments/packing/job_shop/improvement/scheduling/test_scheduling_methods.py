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

from jumanji.environments.packing.job_shop.improvement.scheduling import (
    FlowDueDateMostWorkMethod,
    MethodRegistry,
    PriorityListMethod,
    ShortestProcessingTimeMethod,
)
from jumanji.environments.packing.job_shop.improvement.types import SchedulingMethod


# Test data configurations
@pytest.fixture(
    params=[
        # Standard problem with partial padding
        (
            jnp.array(
                [
                    [0, 1, 2],  # Job 0: machines 0->1->2
                    [1, 0, 2],  # Job 1: machines 1->0->2
                    [1, -1, -1],  # Job 2: only 1 operation
                ]
            ),
            jnp.array(
                [
                    [3, 2, 4],  # Job 0: durations 3,2,4
                    [1, 5, 2],  # Job 1: durations 1,5,2
                    [1, -1, -1],  # Job 2: duration 1
                ]
            ),
        ),
        # Problem with 3rd job padded (only 2 jobs)
        (
            jnp.array(
                [
                    [0, 1, 2],  # Job 0: machines 0->1->2
                    [1, 0, 2],  # Job 1: machines 1->0->2
                    [-1, -1, -1],  # Job 2: no operations (full padding)
                ]
            ),
            jnp.array(
                [
                    [3, 2, 4],  # Job 0: durations 3,2,4
                    [1, 5, 2],  # Job 1: durations 1,5,2
                    [-1, -1, -1],  # Job 2: no operations
                ]
            ),
        ),
    ],
    ids=["partial_padding", "full_padding"],
)
def sample_problem(request: pytest.FixtureRequest) -> tuple[chex.Array, chex.Array]:
    """Create test problems with different padding scenarios."""
    return request.param  # type: ignore[no-any-return]


class TestPriorityListMethod:
    """Test priority list scheduling method."""

    def test_method_properties(self) -> None:
        """Test basic method properties."""
        method = PriorityListMethod(num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3)
        assert method.name == "priority_list"


class TestShortestProcessingTimeMethod:
    """Test shortest processing time scheduling method."""

    def test_method_properties(self) -> None:
        """Test basic method properties."""
        method = ShortestProcessingTimeMethod(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        assert method.name == "shortest_processing_time"


class TestFlowDueDateMostWorkMethod:
    """Test flow due date/most work remaining scheduling method."""

    def test_method_properties(self) -> None:
        """Test basic method properties."""
        method = FlowDueDateMostWorkMethod(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        assert method.name == "flow_due_date_most_work"


class TestMethodRegistry:
    """Test scheduling method registry functionality."""

    def test_all_methods_registered(self) -> None:
        """Test that all concrete methods are properly registered."""
        # Get instances of all methods
        priority_method = PriorityListMethod(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        spt_method = ShortestProcessingTimeMethod(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        fdd_method = FlowDueDateMostWorkMethod(
            num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )

        # Verify they can be retrieved from registry
        assert isinstance(
            MethodRegistry.get_method(
                priority_method.name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
            ),
            PriorityListMethod,
        )
        assert isinstance(
            MethodRegistry.get_method(
                spt_method.name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
            ),
            ShortestProcessingTimeMethod,
        )
        assert isinstance(
            MethodRegistry.get_method(
                fdd_method.name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
            ),
            FlowDueDateMostWorkMethod,
        )

    def test_get_method_names(self) -> None:
        """Test that all expected methods are registered."""
        methods = MethodRegistry.get_method_names()
        expected_methods = {
            "priority_list",
            "shortest_processing_time",
            "flow_due_date_most_work",
        }
        assert set(methods) == expected_methods

    def test_get_method_instance(self) -> None:
        """Test method instance retrieval."""
        method = MethodRegistry.get_method(
            "priority_list", num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        assert hasattr(method, "name")
        assert method.name == "priority_list"

    def test_unknown_method_raises_error(self) -> None:
        """Test that unknown method names raise ValueError."""
        with pytest.raises(ValueError, match="Method 'unknown' not found"):
            MethodRegistry.get_method(
                "unknown", num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
            )

    def test_registry_order_matches_enum_order(self) -> None:
        """Test that the order of methods in registry matches SchedulingMethod enum order.

        This is critical for calls that use enum values as indices
        into method lists created from the registry.
        """
        # Get the expected names based on enum order
        expected_names = []
        for enum_val in SchedulingMethod:
            # Get instances to check which method corresponds to each enum value
            instances = MethodRegistry.get_instances(3, 3, 3, 3)
            method_instance = instances[enum_val.value]
            expected_names.append(method_instance.name)

        # Test 1: Registry order should match the order when accessing instances by enum value
        instances = MethodRegistry.get_instances(3, 3, 3, 3)
        actual_names = [instance.name for instance in instances]

        assert actual_names == expected_names, (
            f"Registry order doesn't match SchedulingMethod enum order!\n"
            f"Registry order: {actual_names}\n"
            f"Expected order: {expected_names}\n"
            f"This will cause jax.lax.switch(method, branches) to call the wrong method!"
        )


class TestBasicSchedulingProperties:
    """Test basic properties that all scheduling methods should satisfy."""

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_matrix_shape(self, method_name: str, sample_problem: tuple) -> None:
        """Test that all methods return correctly shaped matrices."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # Matrix should be square
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        expected_size = 3 * 3 + 2  # num_jobs * max_num_ops + 2 (source + target)
        assert adj_matrix.shape == (expected_size, expected_size)

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_non_negative_weights(self, method_name: str, sample_problem: tuple) -> None:
        """Test that all methods return non-negative edge weights."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # All edge weights should be non-negative
        assert jnp.all(adj_matrix >= 0)

    @pytest.mark.parametrize("method_name", ["shortest_processing_time", "flow_due_date_most_work"])
    def test_deterministic_methods(self, method_name: str, sample_problem: tuple) -> None:
        """Test that deterministic methods produce identical results."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )

        result1 = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=None)
        result2 = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=None)

        # Deterministic methods should return identical results
        assert jnp.allclose(result1, result2)

    def test_priority_list_deterministic_with_same_key(self, sample_problem: tuple) -> None:
        """Test that priority list method is deterministic with same key."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            "priority_list", num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42)

        result1 = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)
        result2 = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # Same key should produce identical results
        assert jnp.allclose(result1, result2)

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_no_self_loops(self, method_name: str, sample_problem: tuple) -> None:
        """Test that adjacency matrices have no self-loops."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # No self-loops (diagonal should be zero)
        assert jnp.all(jnp.diag(adj_matrix) == 0)

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_no_cycles_in_machine_constraints(
        self, method_name: str, sample_problem: tuple
    ) -> None:
        """Test that machine adjacency matrix has no cycles."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # Test for cycles using matrix powers - if A^n has non-zero diagonal, there are cycles
        n = adj_matrix.shape[0]
        current_power = adj_matrix
        for _ in range(n):
            # Check diagonal of current power
            diagonal = jnp.diag(current_power)
            # Only check non-zero entries (where cycles could exist)
            cycle_detected = jnp.any(diagonal > 0)
            assert not cycle_detected, f"Cycle detected in adjacency matrix for {method_name}"
            # Compute next power for next iteration
            current_power = current_power @ adj_matrix

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_operation_connectivity_constraints(
        self, method_name: str, sample_problem: tuple
    ) -> None:
        """Test that each operation is connected to at most one predecessor and one successor."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # Convert to binary matrix (>0 means edge exists)
        binary_matrix = adj_matrix > 0

        # Each operation (row) should have at most one outgoing edge
        outgoing_edges = jnp.sum(binary_matrix, axis=1)
        assert jnp.all(outgoing_edges <= 1), "Operation has more than one successor"

        # Each operation (column) should have at most one incoming edge
        incoming_edges = jnp.sum(binary_matrix, axis=0)
        assert jnp.all(incoming_edges <= 1), "Operation has more than one predecessor"

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_source_target_isolation(self, method_name: str, sample_problem: tuple) -> None:
        """Test that source and target nodes are properly isolated."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)

        # Source node (index 0) should have no incoming edges
        assert jnp.sum(adj_matrix[:, 0]) == 0, "Source node has incoming edges"

        # Target node (last index) should have no outgoing edges
        target_idx = adj_matrix.shape[0] - 1
        assert jnp.sum(adj_matrix[target_idx, :]) == 0, "Target node has outgoing edges"

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_invalid_operations_isolation(self, method_name: str, sample_problem: tuple) -> None:
        """Test that invalid operations (with -1 values) are not connected."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        adj_matrix = method.build_machine_adjacency_matrix(ops_machine_ids, ops_durations, key=key)
        jax.debug.print("adj_matrix: {adj_matrix}", adj_matrix=adj_matrix)
        # Find invalid operations (those with machine_id = -1)
        max_num_jobs, max_num_ops = ops_machine_ids.shape

        for job_id in range(max_num_jobs):
            for op_id in range(max_num_ops):
                if ops_machine_ids[job_id, op_id] == -1:
                    # This is an invalid operation
                    node_idx = job_id * max_num_ops + op_id + 1  # +1 for source node offset

                    # Invalid operation should have no outgoing edges
                    assert (
                        jnp.sum(adj_matrix[node_idx, :]) == 0
                    ), f"Invalid operation at job {job_id}, op {op_id} has outgoing edges"

                    # Invalid operation should have no incoming edges (except possibly from source)
                    incoming_edges = jnp.sum(adj_matrix[:, node_idx])
                    if incoming_edges > 0:
                        # Only source should connect to invalid operations,
                        # and only if it's the first op
                        assert adj_matrix[0, node_idx] > 0 and op_id == 0, (
                            f"Invalid operation at job {job_id}, op {op_id} "
                            f"has non-source incoming edges"
                        )


class TestJitCompatibility:
    """Test JAX JIT compatibility for all scheduling methods."""

    @pytest.mark.parametrize(
        "method_name", ["priority_list", "shortest_processing_time", "flow_due_date_most_work"]
    )
    def test_method_jittability(self, method_name: str, sample_problem: tuple) -> None:
        """Test that all methods are JIT-compatible and compile only once."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )
        key = jax.random.PRNGKey(42) if method_name == "priority_list" else None

        # Clear trace counter and create JIT-compiled function
        chex.clear_trace_counter()
        jit_fn = jax.jit(chex.assert_max_traces(method.build_machine_adjacency_matrix, n=1))

        # First call
        result1 = jit_fn(ops_machine_ids, ops_durations, key=key)

        # Second call should use cached compilation
        result2 = jit_fn(ops_machine_ids, ops_durations, key=key)

        # Results should be identical for deterministic methods
        if method_name != "priority_list":
            assert jnp.allclose(result1, result2)
        else:
            # For priority list with same key, should still be identical
            assert jnp.allclose(result1, result2)

    @pytest.mark.parametrize("method_name", ["shortest_processing_time", "flow_due_date_most_work"])
    def test_deterministic_methods_jit(self, method_name: str, sample_problem: tuple) -> None:
        """Test that deterministic methods produce identical results under JIT."""
        ops_machine_ids, ops_durations = sample_problem
        method = MethodRegistry.get_method(
            method_name, num_jobs=3, num_machines=3, max_num_jobs=3, max_num_ops=3
        )

        # Clear trace counter and JIT compile the function
        chex.clear_trace_counter()
        jit_fn = jax.jit(chex.assert_max_traces(method.build_machine_adjacency_matrix, n=1))

        # Multiple calls should give identical results
        result1 = jit_fn(ops_machine_ids, ops_durations, key=None)
        result2 = jit_fn(ops_machine_ids, ops_durations, key=None)
        result3 = jit_fn(ops_machine_ids, ops_durations, key=None)

        # All results should be identical
        assert jnp.allclose(result1, result2)
        assert jnp.allclose(result2, result3)
        assert jnp.allclose(result1, result3)

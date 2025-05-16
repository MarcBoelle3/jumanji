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

# -----------------------------------------------------------------------------
# This implementation is inspired by the paper:
# "Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling"
# by Cong Zhang, Zhiguang Cao, Wen Song, Yaoxin Wu, Jie Zhang
# (https://arxiv.org/abs/2211.10936).
#
# The logic implemented here corresponds to the mechanism described in
# Section 4.4: "Message Passing for Calculating Schedule".
# Based on the adjacency matrix of the disjunctive graph, it allows to compute
# earliest, latest start times and makespan for a given schedule.
# For clarity, the notations used are the ones of the paper.
# -----------------------------------------------------------------------------

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jraph


def update_node_fn(
    nodes: chex.Array,
    _sent_attributes: chex.Array,
    received_attributes: chex.Array,
    _globals: chex.Array,
) -> chex.Array:
    """Update the date and completion status of a node based on the received attributes.
       Used both in forward and backward pass.

    Args:
        nodes: Array of nodes with shape (num_nodes, 3)
        _sent_attributes: Array of sent attributes with shape (num_edges, 2)
        received_attributes: Array of received attributes with shape (num_edges, 2)
        _globals: Array of globals with shape (num_globals, 1)
    Returns:
        Array of updated nodes with shape (num_nodes, 3)
    """

    # Each node has a feature (duration, start_time, completion) that updates accross
    # message passing:
    # duration the processing time (float)
    # start_time the current start time of the operation (float)
    # completion the current completion status (0 when completed, 1 when not completed)

    # The duration of the operation remains unchanged
    node_duration = nodes[:, 0]

    # Update start_time and completion status from incoming edges
    # received_attributes contains (start_time, completion) of the sender node
    received_date = received_attributes[:, 0]
    received_completion = received_attributes[:, 1]

    return jnp.stack([node_duration, received_date, received_completion], axis=-1)


def compute_earliest_start_times_and_makespan(
    adj_mat: chex.Array, ops_durations: chex.Array, max_num_edges: jnp.int32
) -> Tuple[chex.Array, jnp.float32]:
    """Compute earliest start times and makespan using jraph message passing with max pooling.
       It is an adaptation for GPU of the traditional CPM (Critical Path Method) which computes the
       starting times of the operations recursively.When starting, the source node is the only one
       to be scheduled, and message passing propagates the start times recursively to the target
       node, which in the end contains the makespan.
       First part of the algorithm in Section 4.4 of the paper (https://arxiv.org/abs/2211.10936).


    Args:
        adj_mat: Adjacency matrix of the disjunctive graph
                 shape (max_num_jobs x max_num_ops + 2, max_num_jobs x max_num_ops + 2)
        ops_durations: Processing times of the operations (max_num_jobs, max_num_ops)
        max_num_edges: Maximum number of edges in the graph.

    Returns:
        Array of earliest start times for each node and total makespan.
        Earliest start times correspond, for each operation, to start the earliest possible
        (as soon as the machine is available).
    """

    num_nodes = adj_mat.shape[0]

    # Convert ops_durations to a vector of size num_nodes
    ops_durations = ops_durations.reshape(-1)
    # Add sink and target nodes with duration of 0
    ops_durations = jnp.concatenate([jnp.array([0.0]), ops_durations, jnp.array([0.0])])

    # Create node features: (duration, start_time, completion)
    node_features = jnp.zeros((num_nodes, 3))

    # Initialize completion status to 1 (no completion) except for the source node
    node_features = node_features.at[:, 2].set(jnp.ones(num_nodes))
    node_features = node_features.at[0].set(jnp.array([0.0, 0.0, 0.0]))  # Source node
    # Set operation durations (0 for start and target nodes)
    node_features = node_features.at[:, 0].set(ops_durations)

    # Dummy initialization of edge features
    # In jraph, edge features are used to propagate information between nodes
    # We will use them to propagate the start times and completion status
    edge_features = jnp.zeros((max_num_edges, 2))

    senders, receivers = jnp.nonzero(adj_mat > 0, size=max_num_edges, fill_value=0)

    # Create graph
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(senders)]),
        globals=None,
    )

    def update_edge_fn(
        _edges: chex.Array,
        sender_features: chex.Array,
        _receiver_features: chex.Array,
        _globals: chex.Array,
    ) -> chex.Array:
        """Update the edge features based on the node features (only the sender features are used).
           Due to jraph framework, all args are needed but only sender_features is used.

        Args:
            _edges: Array of edge features with shape (num_edges, 2). Not used.
            sender_features: Array of sender features with shape (num_edges, 3) containing:
                - duration of the operation (float)
                - current start_time of the operation (float)
                - current completion status (0 or 1)
            _receiver_features: Not used.
            _globals: Not used.

        Returns:
            Array of updated edge features with shape (num_edges, 2)
        """

        sender_duration = sender_features[:, 0]
        sender_start_time = sender_features[:, 1]
        sender_completion = sender_features[:, 2]

        new_start_time = sender_start_time + (1.0 - sender_completion) * sender_duration
        new_completion = sender_completion

        return jnp.stack([new_start_time, new_completion], axis=-1)

    # Create message passing layer
    net = jraph.GraphNetwork(
        update_node_fn=update_node_fn,
        update_edge_fn=update_edge_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_max,
        aggregate_nodes_for_globals_fn=None,
        aggregate_edges_for_globals_fn=None,
    )

    def check_completion(graph: jraph.GraphsTuple) -> jnp.bool_:
        """Check if the target node is completed.
           Used as a stopping condition in the while loop.

        Args:
            graph: Current graph state.

        Returns:
            True if the target node is completed, False otherwise.
        """
        return graph.nodes[-1, 2] == 1

    def update_graph(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Update the graph using the message passing layer.

        Args:
            graph: Current graph state.

        Returns:
            Updated graph state.
        """
        output_graph = net(graph)
        # Keep source node to (0, 0, 0)
        output_graph = output_graph._replace(
            nodes=output_graph.nodes.at[0].set(jnp.array([0.0, 0.0, 0.0]))
        )
        return output_graph

    # Run the message passing loop until the target node is completed
    output_graph = jax.lax.while_loop(check_completion, update_graph, graph)

    # Return earliest start times and makespan
    return output_graph.nodes[:, 1], output_graph.nodes[-1, 1]


def compute_latest_start_times(
    adj_mat: chex.Array, ops_durations: chex.Array, makespan: jnp.float32, max_num_edges: jnp.int32
) -> chex.Array:
    """Compute latest start times using message passing with max pooling.
       It is the same method as the forward pass but with reversed edges and propagation from target
       to source, to obtain the latest start times. The latest start times of operations are the
       latest possible starting times for every operation such that the makespan is not exceeded.
       Second part of the algorithm in Section 4.4 of the paper (https://arxiv.org/abs/2211.10936).

    Args:
        adj_mat: Adjacency matrix of the disjunctive graph
                 shape (max_num_jobs x max_num_ops + 2, max_num_jobs x max_num_ops + 2)
        ops_durations: Processing times of the operations (max_num_jobs, max_num_ops)
        makespan: Makespan of the schedule obtained from the forward pass (float)
        max_num_edges: Maximum number of edges in the graph.

    Returns:
        Array of latest start times for each node.
        Latest start times correspond, for each operation, to start the latest possible
        so that it does not exceed the makespan.
    """
    num_nodes = adj_mat.shape[0]

    # convert ops_durations to a vector of size num_nodes
    ops_durations = ops_durations.reshape(-1)
    # add sink and target nodes with duration of 0
    ops_durations = jnp.concatenate([jnp.array([0.0]), ops_durations, jnp.array([0.0])])

    # Create node features: (duration, negative_start_time, completion)
    node_features = jnp.ones((num_nodes, 3))
    # Initialize start times to -1 except for the target node
    # (which has negative start time = - makespan)
    node_features = node_features.at[:, 1].set(-jnp.ones(num_nodes))
    node_features = node_features.at[-1].set(jnp.array([0.0, -makespan, 0.0]))

    # Set operation durations (0 for start and target nodes)
    node_features = node_features.at[:, 0].set(ops_durations)

    # Transposed matrix to get reversed edges
    senders, receivers = jnp.nonzero(adj_mat.T > 0, size=max_num_edges, fill_value=num_nodes - 1)

    # Dummy initialization of edge features
    edge_features = jnp.zeros((max_num_edges, 2))

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(senders)]),
        globals=None,
    )

    def update_edge_fn(
        _edges: chex.Array,
        sender_features: chex.Array,
        _receiver_features: chex.Array,
        _globals: chex.Array,
    ) -> chex.Array:
        """Update the edge features based on the node features (only the sender features are used).
           Due to jraph framework, all args are needed but only sender_features is used.

        Args:
            _edges: Array of edge features with shape (num_edges, 2). Not used.
            sender_features: Array of sender features with shape (num_edges, 3) containing:
                - duration of the operation (float)
                - current negative start time of the operation (float)
                - current completion status (0 or 1)
            _receiver_features: Not used.
            _globals: Not used.

        Returns:
            Array of updated edge features with shape (num_edges, 2)
        """

        sender_duration = _receiver_features[:, 0]
        sender_start_time = sender_features[:, 1]
        sender_completion = sender_features[:, 2]

        new_negative_start_time = sender_start_time + (1.0 - sender_completion) * sender_duration
        new_completion = sender_completion

        return jnp.stack([new_negative_start_time, new_completion], axis=-1)

    net = jraph.GraphNetwork(
        update_node_fn=update_node_fn,
        update_edge_fn=update_edge_fn,
        aggregate_edges_for_nodes_fn=jraph.segment_max,
        aggregate_nodes_for_globals_fn=None,
        aggregate_edges_for_globals_fn=None,
    )

    def check_completion(graph: jraph.GraphsTuple) -> jnp.bool_:
        """Check if the source node is completed.
           Used as a stopping condition in the while loop.

        Args:
            graph: Current graph state.

        Returns:
            True if the source node is completed, False otherwise.
        """
        return graph.nodes[0, 2] == 1

    def update_graph(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Update the graph using the message passing layer.

        Args:
            graph: Current graph state.

        Returns:
            Updated graph state.
        """
        output_graph = net(graph)
        # Keep target node to (0, -makespan, 0)
        output_graph = output_graph._replace(
            nodes=output_graph.nodes.at[-1].set(jnp.array([0.0, -makespan, 0.0]))
        )
        return output_graph

    # Run the message passing loop until the source node is completed
    output_graph = jax.lax.while_loop(check_completion, update_graph, graph)

    # Return latest start times
    return -output_graph.nodes[:, 1]


def compute_est_lst_makespan(
    adj_mat: chex.Array, ops_durations: chex.Array, max_num_edges: jnp.int32
) -> Tuple[chex.Array, chex.Array, jnp.float32]:
    """Compute earliest, latest start times and makespan.

    Args:
        adj_mat: Adjacency matrix of the disjunctive graph
                 shape (max_num_jobs x max_num_ops + 2, max_num_jobs x max_num_ops + 2)
        ops_durations: Processing times of the operations (max_num_jobs, max_num_ops)
    Returns:
        est: earliest start times (max_num_jobs x max_num_ops,)
        lst: latest start times (max_num_jobs x max_num_ops,)
        makespan: makespan of the schedule (float)
    """

    est, makespan = compute_earliest_start_times_and_makespan(adj_mat, ops_durations, max_num_edges)
    lst = compute_latest_start_times(adj_mat, ops_durations, makespan, max_num_edges)
    return est, lst, makespan

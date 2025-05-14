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

    Args:
        nodes: Array of nodes with shape (num_nodes, 3)
        _sent_attributes: Array of sent attributes with shape (num_edges, 2)
        received_attributes: Array of received attributes with shape (num_edges, 2)
        _globals: Array of globals with shape (num_globals, 1)
    Returns:
        Array of updated nodes with shape (num_nodes, 3)
    """

    # Each node has a feature (p, d, c) with p the processing time,
    # d the date and c the completion status
    node_p = nodes[:, 0]  # remains unchanged

    # Update date and completion status if neighbours
    received_d = received_attributes[:, 0]
    received_c = received_attributes[:, 1]

    return jnp.stack([node_p, received_d, received_c], axis=-1)


def forward_pass_jraph(
    adj_mat: chex.Array, ops_durations: chex.Array, max_num_edges: jnp.int32
) -> Tuple[chex.Array, jnp.float32]:
    """Compute earliest start times and makespan using jraph message passing with max pooling.
        First part of the algorithm in Section 4.4 of the paper.

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

    # Create node features: (processing time, date, completion status)
    node_features = jnp.zeros((num_nodes, 3))
    # Initialize features as in the paper
    node_features = node_features.at[:, 2].set(jnp.ones(num_nodes))
    node_features = node_features.at[0].set(jnp.array([0.0, 0.0, 0.0]))  # Source node
    node_features = node_features.at[:, 0].set(ops_durations)  # Add processing times

    # Create edge features
    edge_features = jnp.zeros((max_num_edges, 2))  # dummy initialization
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
        """Update the edge features based on the sender and receiver features.

        Args:
            _edges: Array of edge features with shape (num_edges, 2). Not used.
            sender_features: Array of sender features with shape (num_edges, 3)
            _receiver_features: Array of receiver features with shape (num_edges, 3). Not used.
            _globals: Array of globals with shape (num_globals, 1). Not used.

        Returns:
            Array of updated edge features with shape (num_edges, 2)
        """

        sender_p = sender_features[:, 0]
        sender_d = sender_features[:, 1]
        sender_c = sender_features[:, 2]

        new_d = sender_p + (1.0 - sender_c) * sender_d
        new_c = sender_c

        return jnp.stack([new_d, new_c], axis=-1)

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


def backward_pass_jraph(
    adj_mat: chex.Array, ops_durations: chex.Array, makespan: jnp.float32, max_num_edges: jnp.int32
) -> chex.Array:
    """Compute latest start times using message passing with max pooling.
        Second part of the algorithm in Section 4.4 of the paper.

    Args:
        adj_mat: Adjacency matrix of the disjunctive graph
                 shape (max_num_jobs x max_num_ops + 2, max_num_jobs x max_num_ops + 2)
        ops_durations: Processing times of the operations (max_num_jobs, max_num_ops)
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

    # Create node features: (processing time, date, completion status)
    node_features = jnp.ones((num_nodes, 3))
    node_features = node_features.at[:, 1].set(-jnp.ones(num_nodes))
    node_features = node_features.at[-1].set(jnp.array([0.0, -makespan, 0.0]))  # Source node
    node_features = node_features.at[:, 0].set(ops_durations)  # add processing times

    # Create edge features: edge weights
    # Transposed matrix to get reversed edges
    senders, receivers = jnp.nonzero(adj_mat.T > 0, size=max_num_edges, fill_value=num_nodes - 1)

    edge_features = jnp.zeros((max_num_edges, 2))  # dummy initialization

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(senders)]),
        globals=None,
    )
    jax.debug.print("graph created")

    def update_edge_fn(
        _edges: chex.Array,
        sender_features: chex.Array,
        _receiver_features: chex.Array,
        _globals: chex.Array,
    ) -> chex.Array:
        # sender_features: [num_edges, 3]  with (p, d, c)

        sender_p = _receiver_features[:, 0]
        sender_d = sender_features[:, 1]
        sender_c = sender_features[:, 2]

        new_d = sender_p + (1.0 - sender_c) * sender_d
        new_c = sender_c

        return jnp.stack([new_d, new_c], axis=-1)

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
        output_graph = output_graph._replace(
            nodes=output_graph.nodes.at[-1].set(jnp.array([0.0, -makespan, 0.0]))
        )  # keep source node to (0, 0)
        return output_graph

    # Run the message passing loop until the source node is completed
    output_graph = jax.lax.while_loop(check_completion, update_graph, graph)

    # Return latest start times
    return -output_graph.nodes[:, 1]


def forward_backward_pass_jraph(
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

    est, makespan = forward_pass_jraph(adj_mat, ops_durations, max_num_edges)
    lst = backward_pass_jraph(adj_mat, ops_durations, makespan, max_num_edges)
    return est, lst, makespan

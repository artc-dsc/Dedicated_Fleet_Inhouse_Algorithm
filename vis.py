import pandas as pd
import numpy as np
import random

import networkx as nx
import matplotlib.pyplot as plt


# Global Params
class g:
    num_order_max = 6


# Helper Functions
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# Class for OB Graph
class OB_Graph:
    # Init OB Graph instance
    def __init__(self, df):
        self.df = df
        self.nodes_list = ['G']
        self.edges_list = []

    # Function to transform the data and update the nodes & edges lists
    def create_nodes_and_edges(self):

        # Iterates over each row
        for i, row in self.df.iterrows():

            # Keep track of the node names
            prev_id = ''

            # Dfs - goes through all 6 columns
            for n in range(1, g.num_order_max + 1):

                col = 'order_' + str(n)
                current_id = str(row[col])

                # Terminates if reach 0
                if current_id == '0':
                    break

                # Concat to get the new node id
                if col == 'order_1':
                    node_id = prev_id + current_id
                else:
                    node_id = prev_id + ',' + current_id

                ### Update Nodes List ###
                # If node id already exists don't add
                if node_id not in self.nodes_list:
                    self.nodes_list.append(node_id)
                    self.depth_x_nodes_count[n] += 1

                ### Update Edges List ###
                if col == 'order_1':
                    edge_pair = ('G', node_id)
                else:
                    edge_pair = (prev_id, node_id)

                if edge_pair not in self.edges_list:
                    self.edges_list.append(edge_pair)

                prev_id = node_id

    # Function to plot graphs
    def plot(self):
        # Initialise an empty graph
        G = nx.DiGraph()

        # Add nodes from the nodes list
        G.add_nodes_from(self.nodes_list)

        # Add edges from the edges list
        G.add_edges_from(self.edges_list)

        print(f'Is Tree: {nx.is_tree(G)}')

        # Draw graph
        pos2 = hierarchy_pos(G, 'G')
        nx.draw(G, pos2, with_labels=True)
        plt.show()

    # Read Input


# in_df = pd.read_excel('ob_data_test.xlsx', sheet_name='order_bundle')
in_df = pd.read_csv('ob_data.csv')[:20]


# Initialse OB_Graph obj
ob_graph = OB_Graph(in_df)

# Transform data
ob_graph.create_nodes_and_edges()

# Plot graph
ob_graph.plot()












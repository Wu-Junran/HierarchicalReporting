#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com
# date:    2020-11-26 16:09:29

import os
import sys
import copy
import json
import time
import pickle
import itertools
import traceback
import numpy as np
import networkx as nx
from multiprocessing import Pool
from lib.coding_tree import PartitionTree


PWD = os.path.dirname(os.path.realpath(__file__))


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_encoding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id= [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


def pool_trans(input_):
    g, tree_depth = input_
    adj_mat = trans_to_adj(g['G'])
    tree = trans_to_tree(adj_mat, tree_depth)
    g['tree'] = update_node(tree)
    return g


def struct_tree(dataset, tree_depth):
    if not os.path.exists('trees'):
        os.makedirs('trees')
    if os.path.exists('trees/%s_%s.pickle' % (dataset, tree_depth)):
        return
    with open('graphs/%s.pickle' % dataset, 'rb') as fp:
        g_list = pickle.load(fp)
    pool = Pool()
    g_list = pool.map(pool_trans, [(g, tree_depth) for g in g_list])
    pool.close()
    pool.join()
    g_list = filter(lambda g: g is not None, g_list)
    with open('trees/%s_%s.pickle' % (dataset, tree_depth), 'wb') as fp:
        pickle.dump(list(g_list), fp)


if __name__ == '__main__':
    disconnected_dataset = ['REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 'NCI1']
    for d in os.listdir('datasets'):
        if d in disconnected_dataset:
            continue
        for k in [2, 3, 4, 5]:
            print(d, k)
            struct_tree(d[:-4], k)

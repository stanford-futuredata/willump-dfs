from typing import MutableMapping, List

from featuretools.feature_base.feature_base import FeatureBase

from willump_dfs.graph.willump_dfs_graph_node import WillumpDFSGraphNode

import copy


class WillumpDFSGraph(object):

    def __init__(self):
        self._graph_dict: MutableMapping[str, WillumpDFSGraphNode] = {}
        self._top_level_nodes: List[WillumpDFSGraphNode] = []

    def add_new_feature(self, feature: FeatureBase) -> None:

        def make_node_for_feature(feature: FeatureBase) -> WillumpDFSGraphNode:
            if feature.get_name() in self._graph_dict:
                return self._graph_dict[feature.get_name()]
            feature_dependencies: List[FeatureBase] = feature.get_dependencies(deep=False)
            if len(feature_dependencies) == 0:
                feature_node = WillumpDFSGraphNode(feature, None)
                self._graph_dict[feature.get_name()] = feature_node
                return feature_node
            graph_dependencies = list(map(make_node_for_feature, feature_dependencies))
            feature_node = WillumpDFSGraphNode(feature, graph_dependencies)
            self._graph_dict[feature.get_name()] = feature_node
            return feature_node
        feature_node = make_node_for_feature(feature)
        self._top_level_nodes.append(feature_node)
        return

    def __str__(self) -> str:
        return_string = ""
        visited_nodes = set()
        visit_stack = copy.copy(self._top_level_nodes)
        while len(visit_stack) > 0:
            next_node = visit_stack.pop()
            if next_node in visited_nodes:
                continue
            if next_node.get_dependencies() is None:
                visited_nodes.add(next_node)
                continue
            else:
                for dependency in next_node.get_dependencies():
                    return_string += "%s -> %s\n" % (next_node.get_feature(), dependency.get_feature())
                    visit_stack.append(dependency)
                visited_nodes.add(next_node)
        return return_string




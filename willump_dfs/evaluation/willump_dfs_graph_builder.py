from featuretools.feature_base.feature_base import FeatureBase
from typing import List

from willump_dfs.graph.willump_dfs_graph import WillumpDFSGraph


def willump_dfs_build_graph(features: List[FeatureBase]) -> WillumpDFSGraph:
    willump_dfs_graph = WillumpDFSGraph()
    for feature in features:
        willump_dfs_graph.add_new_feature(feature)
    print(willump_dfs_graph)
    return willump_dfs_graph

from typing import List

from featuretools.feature_base.feature_base import FeatureBase

from willump_dfs.graph.willump_dfs_graph import WillumpDFSGraph


def willump_dfs_partition_features(features: List[FeatureBase]) -> List[List[FeatureBase]]:
    """
    Partition a list of features into disjoint sets of features which share no dependencies.
    """
    willump_dfs_graph = WillumpDFSGraph()
    for feature in features:
        willump_dfs_graph.add_new_feature(feature)
    partitioned_features = willump_dfs_graph.partition_features()
    for entry in partitioned_features:
        print(entry)
    return partitioned_features

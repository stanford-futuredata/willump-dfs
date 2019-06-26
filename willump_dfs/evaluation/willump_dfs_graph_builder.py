import time
from typing import List, Tuple

import featuretools as ft
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
    return partitioned_features


def willump_dfs_time_partitioned_features(partitioned_features: List[List[FeatureBase]], validation_entity_set,
                                          validation_label_times) -> List[float]:
    partition_times = []
    for feature_set in partitioned_features:
        t0 = time.time()
        ft.calculate_feature_matrix(feature_set,
                                    entityset=validation_entity_set,
                                    cutoff_time=validation_label_times)
        time_elapsed = time.time() - t0
        partition_times.append(time_elapsed)
    return partition_times


def willump_dfs_get_partition_importances(partitioned_features: List[List[FeatureBase]], features: List[FeatureBase],
                                          feature_importances: List[float]) -> List[float]:
    feature_importance_map = {feature: importance for feature, importance in zip(features, feature_importances)}
    feature_importance_list = []
    for feature_set in partitioned_features:
        partition_importance = 0
        for feature in feature_set:
            partition_importance += feature_importance_map[feature]
        feature_importance_list.append(partition_importance)
    return feature_importance_list


def willump_dfs_find_efficient_features(partitioned_features: List[List[FeatureBase]], partition_times: List[float],
                                        partition_importances: List[float]) \
        -> Tuple[List[FeatureBase], List[FeatureBase]]:
    return [], []

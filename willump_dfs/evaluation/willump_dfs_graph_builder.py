import copy
import time
import numpy as np
import pandas as pd
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
                                          validation_times) -> List[float]:
    partition_times = []
    for feature_set in partitioned_features:
        t0 = time.time()
        ft.calculate_feature_matrix(feature_set,
                                    entityset=validation_entity_set,
                                    cutoff_time=validation_times)
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


def willump_dfs_find_efficient_features(partitioned_features: List[List[FeatureBase]], partition_costs: List[float],
                                        partition_importances: List[float]) \
        -> Tuple[List[FeatureBase], List[FeatureBase]]:
    """
    Implements Willump's algorithm for finding efficient features.
    """
    total_cost = sum(partition_costs)
    partition_ids = range(len(partitioned_features))
    partition_efficiencies = [importance / cost for importance, cost in zip(partition_importances, partition_costs)]
    ranked_partitions = sorted(partition_ids, key=lambda x: partition_efficiencies[x], reverse=True)
    current_cost = 0
    current_importance = 0
    more_important_partitions = []
    for p_id in ranked_partitions:
        if current_cost == 0:
            average_efficiency = 0
        else:
            average_efficiency = current_importance / current_cost
        partition_efficiency = partition_efficiencies[p_id]
        if partition_efficiency < average_efficiency / 5:
            break
        if current_cost + partition_costs[p_id] <= 0.5 * total_cost:
            more_important_partitions.append(p_id)
            current_cost += partition_costs[p_id]
            current_importance += partition_importances[p_id]
    less_important_partitions = [p_id for p_id in partition_ids if p_id not in more_important_partitions]
    more_important_features, less_important_features = [], []
    for p_id in more_important_partitions:
        more_important_features += partitioned_features[p_id]
    for p_id in less_important_partitions:
        less_important_features += partitioned_features[p_id]
    return more_important_features, less_important_features


def willump_dfs_train_models(more_important_features: List[FeatureBase], less_important_features: List[FeatureBase],
                             entity_set, training_times, y_train, model):
    small_model = copy.copy(model)
    full_model = copy.copy(model)
    full_features = more_important_features + less_important_features
    mi_feature_matrix = ft.calculate_feature_matrix(more_important_features,
                                                    entityset=entity_set,
                                                    cutoff_time=training_times)
    mi_feature_matrix = mi_feature_matrix.fillna(0)
    small_model.fit(mi_feature_matrix, y_train)
    full_feature_matrix = ft.calculate_feature_matrix(full_features,
                                                      entityset=entity_set,
                                                      cutoff_time=training_times)
    full_feature_matrix = full_feature_matrix.fillna(0)
    full_model.fit(full_feature_matrix, y_train)
    return small_model, full_model


def willump_dfs_cascade(more_important_features: List[FeatureBase], less_important_features: List[FeatureBase],
                        entity_set, cutoff_times, small_model, full_model, confidence_threshold):
    mi_feature_matrix = ft.calculate_feature_matrix(more_important_features,
                                                    entityset=entity_set,
                                                    cutoff_time=cutoff_times)
    mi_feature_matrix = mi_feature_matrix.fillna(0)
    small_model_probs = small_model.predict_proba(mi_feature_matrix)
    small_model_preds = small_model.classes_.take(np.argmax(small_model_probs, axis=1), axis=0)
    combined_preds = small_model_preds
    small_model_probs = small_model_probs[:, 1]
    mask = np.logical_and(confidence_threshold >= small_model_probs, small_model_probs >= 1 - confidence_threshold)
    cascaded_times = cutoff_times[mask]
    cascaded_mi_matrix = mi_feature_matrix[mask]
    if len(cascaded_times) > 0:
        li_feature_matrix = ft.calculate_feature_matrix(less_important_features,
                                                        entityset=entity_set,
                                                        cutoff_time=cascaded_times)
        li_feature_matrix = li_feature_matrix.fillna(0)
        full_feature_matrix = pd.concat([cascaded_mi_matrix, li_feature_matrix], axis=1)
        full_model_preds = full_model.predict(full_feature_matrix)
        f_index = 0
        for i in range(len(combined_preds)):
            if mask[i]:
                combined_preds[i] = full_model_preds[f_index]
                f_index += 1
    return combined_preds

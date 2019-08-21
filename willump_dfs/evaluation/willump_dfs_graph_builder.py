import random
import time
from typing import List, Tuple

import featuretools as ft
import numpy as np
from featuretools.feature_base.feature_base import FeatureBase
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from willump_dfs.evaluation.willump_dfs_utils import index_feature_in_list
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
                                          validation_times, approximate=None) -> List[float]:
    partition_times = []
    for feature_set in partitioned_features:
        t0 = time.time()
        ft.calculate_feature_matrix(feature_set,
                                    entityset=validation_entity_set,
                                    cutoff_time=validation_times,
                                    approximate=approximate)
        time_elapsed = time.time() - t0
        partition_times.append(time_elapsed)
    return partition_times


def willump_dfs_get_partition_importances(partitioned_features: List[List[FeatureBase]], features: List[FeatureBase],
                                          feature_importances: List[float]) -> List[float]:
    feature_importance_map = {feature: importance for feature, importance in zip(features, feature_importances)}
    partition_importance_list = []
    for feature_set in partitioned_features:
        partition_importance = 0
        for feature in feature_set:
            partition_importance += feature_importance_map[feature]
        partition_importance_list.append(partition_importance)
    return partition_importance_list


def willump_dfs_find_efficient_features(partitioned_features: List[List[FeatureBase]], partition_costs: List[float],
                                        partition_importances: List[float], cost_cutoff: float) \
        -> Tuple[List[FeatureBase], List[FeatureBase], float, float]:
    """
    Implements Willump's algorithm for finding efficient features.
    """
    total_cost = sum(partition_costs)
    partition_ids = range(len(partitioned_features))
    partition_efficiencies = [importance / cost for importance, cost in zip(partition_importances, partition_costs)]
    ranked_partitions = sorted(partition_ids, key=lambda x: partition_efficiencies[x], reverse=True)
    current_cost = 0
    more_important_partitions = []
    for p_id in ranked_partitions:
        if current_cost + partition_costs[p_id] <= cost_cutoff * total_cost:
            more_important_partitions.append(p_id)
            current_cost += partition_costs[p_id]
    less_important_partitions = [p_id for p_id in partition_ids if p_id not in more_important_partitions]
    more_important_features, less_important_features = [], []
    for p_id in more_important_partitions:
        more_important_features += partitioned_features[p_id]
    for p_id in less_important_partitions:
        less_important_features += partitioned_features[p_id]
    return more_important_features, less_important_features, current_cost, total_cost


def calculate_feature_set_performance(x, y, mi_cost: float, total_cost: float, mi_features, all_features,
                                      train_function, predict_function, predict_proba_function, score_function):
    mi_indices = list(map(lambda feature: index_feature_in_list(feature, all_features), mi_features))
    x_train, x_holdout, y_train, y_holdout = train_test_split(x, y, test_size=0.25, random_state=42)
    original_model = train_function(x_train, y_train)
    original_model_holdout_predictions = predict_function(original_model, x_holdout)
    original_score = score_function(y_holdout, original_model_holdout_predictions)
    x_train_small, x_holdout_small = x_train[:, mi_indices], x_holdout[:, mi_indices]
    small_model = train_function(x_train_small, y_train)
    small_confidences = predict_proba_function(small_model, x_holdout_small)
    small_preds = predict_function(small_model, x_holdout_small)
    threshold_to_combined_cost_map = {}
    for cascade_threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        combined_preds = original_model_holdout_predictions.copy()
        num_mi_predicted = 0
        for i in range(len(small_confidences)):
            if small_confidences[i] > cascade_threshold or small_confidences[i] < 1 - cascade_threshold:
                num_mi_predicted += 1
                combined_preds[i] = small_preds[i]
        combined_score = score_function(y_holdout, combined_preds)
        frac_mi_predicted = num_mi_predicted / len(combined_preds)
        combined_cost = frac_mi_predicted * mi_cost + (1 - frac_mi_predicted) * total_cost
        if combined_score > original_score - 0.001:
            threshold_to_combined_cost_map[cascade_threshold] = combined_cost
    best_threshold, best_cost = min(threshold_to_combined_cost_map.items(), key=lambda x: x[1])
    return best_threshold, best_cost


orig_model = None


def calculate_feature_set_performance_topk(x, y, mi_cost: float, total_cost: float, mi_features, all_features,
                                           train_function, predict_proba_function,
                                           top_k_distribution: List[int], valid_size_distribution: List[int]):
    global orig_model
    mi_feature_indices = list(map(lambda feature: index_feature_in_list(feature, all_features), mi_features))
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.5, random_state=42)
    assert (valid_x.shape[0] > max(valid_size_distribution))
    train_x_efficient, valid_x_efficient = train_x[:, mi_feature_indices], valid_x[:, mi_feature_indices]
    if orig_model is None:
        orig_model = train_function(train_x, train_y)
    small_model = train_function(train_x_efficient, train_y)
    small_probs = predict_proba_function(small_model, valid_x_efficient)
    orig_probs = predict_proba_function(orig_model, valid_x)
    num_samples = 100
    candidate_ratios = sorted(list(set(range(1, 100)).union(set(range(1, min(valid_size_distribution) //
                                                                      max(top_k_distribution), 10)))))
    ratios_map = {i: 0 for i in candidate_ratios}
    for i in range(num_samples):
        valid_size = random.choice(valid_size_distribution)
        top_k = random.choice(top_k_distribution)
        sample_indices = np.random.choice(len(orig_probs), size=valid_size)
        sample_small_probs = small_probs[sample_indices]
        sample_orig_probs = orig_probs[sample_indices]
        assert (len(sample_small_probs) == len(sample_orig_probs) == valid_size)
        orig_model_top_k_idx = np.argsort(sample_orig_probs)[-1 * top_k:]
        for ratio in candidate_ratios:
            small_model_top_ratio_k_idx = np.argsort(sample_small_probs)[-1 * top_k * ratio:]
            small_model_precision = len(np.intersect1d(orig_model_top_k_idx, small_model_top_ratio_k_idx)) / top_k
            if small_model_precision >= 0.95:
                ratios_map[ratio] += 1
    good_ratios = [ratio for ratio in candidate_ratios if ratios_map[ratio] >= 0.95 * num_samples]
    good_ratio = min(good_ratios)
    cost = mi_cost * np.mean(valid_size_distribution) + (total_cost - mi_cost) * good_ratio *\
           np.mean(top_k_distribution)
    return good_ratio, cost


def willump_dfs_train_models(more_important_features: List[FeatureBase], less_important_features: List[FeatureBase],
                             entity_set, training_times, y_train, train_function, approximate=None):
    full_features = more_important_features + less_important_features
    mi_feature_matrix = ft.calculate_feature_matrix(more_important_features,
                                                    entityset=entity_set,
                                                    cutoff_time=training_times,
                                                    approximate=approximate)
    mi_feature_matrix = mi_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(mi_feature_matrix.median())
    small_model = train_function(mi_feature_matrix, y_train)
    full_feature_matrix = ft.calculate_feature_matrix(full_features,
                                                      entityset=entity_set,
                                                      cutoff_time=training_times,
                                                      approximate=approximate)
    full_feature_matrix = full_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(full_feature_matrix.median())
    full_model = train_function(full_feature_matrix, y_train)
    return small_model, full_model


def willump_dfs_cascade(more_important_features: List[FeatureBase], less_important_features: List[FeatureBase],
                        entity_set, cutoff_times, small_model, full_model, confidence_threshold):
    sort_col = "__willump_sort_col"
    cutoff_times[sort_col] = range(len(cutoff_times))
    mi_feature_matrix = ft.calculate_feature_matrix(more_important_features,
                                                    entityset=entity_set,
                                                    cutoff_time=cutoff_times).sort_values(by=sort_col).drop(sort_col,
                                                                                                            axis=1)
    mi_feature_matrix = mi_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(mi_feature_matrix.median())
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
                                                        cutoff_time=cascaded_times).sort_values(by=sort_col).drop(
            sort_col, axis=1)
        li_feature_matrix = li_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
            fillna(li_feature_matrix.median())
        full_feature_matrix = np.hstack((cascaded_mi_matrix, li_feature_matrix))
        full_model_preds = full_model.predict(full_feature_matrix)
        f_index = 0
        for i in range(len(combined_preds)):
            if mask[i]:
                combined_preds[i] = full_model_preds[f_index]
                f_index += 1
    return combined_preds


def willump_dfs_topk_cascade(more_important_features: List[FeatureBase], less_important_features: List[FeatureBase],
                             entity_set, cutoff_times, small_model, full_model, ratio, top_k):
    sort_col = "__willump_sort_col"
    cutoff_times[sort_col] = range(len(cutoff_times))
    mi_feature_matrix = ft.calculate_feature_matrix(more_important_features,
                                                    entityset=entity_set,
                                                    cutoff_time=cutoff_times).sort_values(by=sort_col).drop(sort_col,
                                                                                                            axis=1)
    mi_feature_matrix = mi_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(mi_feature_matrix.median())
    small_model_probs = small_model.predict_proba(mi_feature_matrix)[:, 1]
    small_model_top_ratio_k_idx = np.argsort(small_model_probs)[-1 * top_k * ratio:]
    mask = np.zeros(len(small_model_probs), dtype=bool)
    mask[small_model_top_ratio_k_idx] = True
    cascaded_times = cutoff_times[mask]
    cascaded_mi_matrix = mi_feature_matrix[mask]
    assert (sum(mask) == top_k * ratio == len(cascaded_times))
    preds = np.zeros(len(small_model_probs))
    li_feature_matrix = ft.calculate_feature_matrix(less_important_features,
                                                    entityset=entity_set,
                                                    cutoff_time=cascaded_times).sort_values(by=sort_col).drop(
        sort_col, axis=1)
    li_feature_matrix = li_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(li_feature_matrix.median())
    full_feature_matrix = np.hstack((cascaded_mi_matrix, li_feature_matrix))
    full_model_preds = full_model.predict_proba(full_feature_matrix)[:, 1]
    f_index = 0
    for i in range(len(preds)):
        if mask[i]:
            preds[i] = full_model_preds[f_index]
            f_index += 1
    return preds


def willump_dfs_permutation_importance(features: List[FeatureBase],
                                       partitioned_features: List[List[FeatureBase]], X, y,
                                       train_function, predict_function, scoring_function) -> List[float]:
    """
    Calculate mean decrease accuracy for every partition.  This is the decrease in OOB accuracy when all features
    in the partition are shuffled.
    """
    partition_indices = list(
        map(lambda partition: list(map(lambda feature: index_feature_in_list(feature, features), partition)),
            partitioned_features))
    scores = [[] for _ in range(len(partition_indices))]
    rs = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    for train_index, test_index in rs.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = train_function(X_train, y_train)
        y_pred = predict_function(model, X_test)
        base_accuracy = scoring_function(y_test, y_pred)
        for i, feature_indices in enumerate(partition_indices):
            for feature_index in feature_indices:
                X_test_copy = X_test.copy()
                np.random.shuffle(X_test_copy[:, feature_index])
                y_pred = predict_function(model, X_test_copy)
                shuffle_accuracy = scoring_function(y_test, y_pred)
                scores[i].append(base_accuracy - shuffle_accuracy)
    return list(map(np.average, scores))

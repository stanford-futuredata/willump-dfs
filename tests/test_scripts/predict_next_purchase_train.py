import argparse
import pickle

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import predict_next_purchase_utils as utils
from willump_dfs.evaluation.willump_dfs_graph_builder import *
from willump_dfs.evaluation.willump_dfs_utils import feature_in_list

resources_folder = "tests/test_resources/predict_next_purchase_resources/"

data_small = "data_small"
data_large = "data_large"
data_full = "data_huge"

data_folder = None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Cascade threshold")
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "small":
        data_folder = data_small
    elif dataset == "large":
        data_folder = data_large
    elif dataset == "huge":
        data_folder = data_full
    else:
        print("Invalid dataset")
        exit(1)

    try:
        es = ft.read_entityset(resources_folder + data_folder + "_entity_set")
    except AssertionError:
        es = utils.load_entityset(resources_folder + data_folder)
        es.to_pickle(resources_folder + data_folder + "_entity_set")

    label_times = utils.make_labels(es=es,
                                    product_name="Banana",
                                    cutoff_time=pd.Timestamp('March 15, 2015'),
                                    prediction_window=ft.Timedelta("4 weeks"),
                                    training_window=ft.Timedelta("60 days"))

    feature_matrix, features = ft.dfs(target_entity="users",
                                      cutoff_time=label_times,
                                      training_window=ft.Timedelta("60 days"),  # same as above
                                      entityset=es,
                                      verbose=False)
    # encode categorical values
    fm_encoded, features_encoded = ft.encode_features(feature_matrix,
                                                      features)

    X = fm_encoded.reset_index().merge(label_times)
    X.drop(["user_id", "time"], axis=1, inplace=True)
    X = X.fillna(0)
    y = X.pop("label")

    clf = utils.pnp_train_function(X, y)
    features_encoded = utils.feature_importances(clf, features_encoded, n=20)

    label_times_train, label_times_test = train_test_split(label_times, test_size=0.2, random_state=42)
    label_times_train = label_times_train.sort_values(by=["user_id"])
    label_times_test = label_times_test.sort_values(by=["user_id"])
    y_train = label_times_train.pop("label")
    y_test = label_times_test.pop("label")

    # Train model with top features.
    top_feature_matrix_train = ft.calculate_feature_matrix(features_encoded,
                                                           entityset=es,
                                                           cutoff_time=label_times_train)
    top_feature_matrix_train = top_feature_matrix_train.fillna(0)

    partitioned_features = willump_dfs_partition_features(features_encoded)

    partition_times = willump_dfs_time_partitioned_features(partitioned_features, es, label_times)
    partition_importances = \
        willump_dfs_permutation_importance(features_encoded, partitioned_features,
                                           top_feature_matrix_train.values, y_train.values,
                                           train_function=utils.pnp_train_function,
                                           predict_function=utils.pnp_predict_function,
                                           scoring_function=roc_auc_score)

    min_cost = np.inf
    more_important_features, less_important_features, cascade_threshold, cost_cutoff = None, None, None, None
    for cc_candidate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        mi_features_candidate, li_features_candidate, mi_cost, total_cost = \
            willump_dfs_find_efficient_features(partitioned_features,
                                                partition_costs=partition_times,
                                                partition_importances=partition_importances, cost_cutoff=cc_candidate)
        t_candidate, cost = calculate_feature_set_performance(x=top_feature_matrix_train.values, y=y_train.values,
                                                              mi_cost=mi_cost, total_cost=total_cost,
                                                              mi_features=mi_features_candidate,
                                                              all_features=features_encoded,
                                                              train_function=utils.pnp_train_function,
                                                              predict_function=utils.pnp_predict_function,
                                                              predict_proba_function=utils.pnp_predict_proba_function,
                                                              score_function=roc_auc_score)
        if cost < min_cost:
            more_important_features = mi_features_candidate
            less_important_features = li_features_candidate
            cascade_threshold = t_candidate
            cost_cutoff = cc_candidate
            min_cost = cost

    print("Cost Cutoff: %f Cascade Threshold: %f" % (cost_cutoff, cascade_threshold))

    for i, (features, cost, importance) in enumerate(zip(partitioned_features, partition_times, partition_importances)):
        print("%d Features: %s\nCost: %f  Importance: %f  Efficient: %r" % (i, features, cost, importance, all(
            feature_in_list(feature, more_important_features) for feature in features)))

    small_model, full_model = willump_dfs_train_models(more_important_features=more_important_features,
                                                       less_important_features=less_important_features,
                                                       entity_set=es,
                                                       training_times=label_times_train,
                                                       y_train=y_train,
                                                       train_function=utils.pnp_train_function)

    # Save top features.
    ft.save_features(less_important_features, resources_folder + "li_features.dfs")
    pickle.dump(full_model, open(resources_folder + "full_model.pk", "wb"))
    ft.save_features(more_important_features, resources_folder + "mi_features.dfs")
    pickle.dump(small_model, open(resources_folder + "small_model.pk", "wb"))
    pickle.dump(cascade_threshold, open(resources_folder + "cascades_parameters.pk", "wb"))

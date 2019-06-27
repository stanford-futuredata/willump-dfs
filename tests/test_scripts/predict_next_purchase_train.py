import pickle
import time

import featuretools as ft
import pandas as pd
import predict_next_purchase_utils as utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

from willump_dfs.evaluation.willump_dfs_graph_builder import *

resources_folder = "tests/test_resources/predict_next_purchase_resources/"

data_small = "data_small"
data_large = "data_large"

data_folder = data_large

if __name__ == '__main__':

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

    clf = RandomForestClassifier(n_estimators=400, n_jobs=1)
    # Select top features.
    clf.fit(X, y)
    top_features = utils.feature_importances(clf, features_encoded, n=100)

    label_times_train, label_times_test = train_test_split(label_times, test_size=0.2, random_state=42)
    label_times_train = label_times_train.sort_values(by=["user_id"])
    label_times_test = label_times_test.sort_values(by=["user_id"])

    # Train model with top features.
    top_feature_matrix_train = ft.calculate_feature_matrix(top_features,
                                                           entityset=es,
                                                           cutoff_time=label_times_train)
    y_train = top_feature_matrix_train.pop("label")
    top_feature_matrix_train = top_feature_matrix_train.fillna(0)

    clf.fit(top_feature_matrix_train, y_train)
    top_feature_importances = clf.feature_importances_

    partitioned_features = willump_dfs_partition_features(top_features)

    partition_times = willump_dfs_time_partitioned_features(partitioned_features, es, label_times)
    partition_importances = willump_dfs_get_partition_importances(partitioned_features, top_features,
                                                                  top_feature_importances)

    more_important_features, less_important_features = \
        willump_dfs_find_efficient_features(partitioned_features,
                                            partition_costs=partition_times,
                                            partition_importances=partition_importances)

    # for feature, cost, importance in zip(partitioned_features, partition_times, partition_importances):
    #     print("Features: %s\nCost: %f  Importance: %f" % (feature, cost, importance))

    small_model, full_model = willump_dfs_train_models(more_important_features=more_important_features,
                                                       less_important_features=less_important_features,
                                                       entity_set=es,
                                                       training_label_times=label_times_train,
                                                       model=clf)

    mi_t0 = time.time()
    mi_feature_matrix_test = ft.calculate_feature_matrix(more_important_features,
                                                         entityset=es,
                                                         cutoff_time=label_times_test)
    y_test = mi_feature_matrix_test.pop("label")
    mi_feature_matrix_test = mi_feature_matrix_test.fillna(0)
    mi_preds = small_model.predict(mi_feature_matrix_test)
    mi_time_elapsed = time.time() - mi_t0
    mi_score = roc_auc_score(y_test, mi_preds)

    full_t0 = time.time()
    full_feature_matrix_test = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                           entityset=es,
                                                           cutoff_time=label_times_test)
    full_feature_matrix_test.drop(["label"], axis=1, inplace=True)
    full_feature_matrix_test = full_feature_matrix_test.fillna(0)
    full_preds = full_model.predict(full_feature_matrix_test)
    full_time_elapsed = time.time() - full_t0
    full_score = roc_auc_score(y_test, full_preds)

    cascade_t0 = time.time()
    cascade_preds = willump_dfs_cascade(more_important_features=more_important_features,
                                        less_important_features=less_important_features,
                                        entity_set=es, label_times=label_times_test, small_model=small_model,
                                        full_model=full_model, confidence_threshold=0.6)
    cascade_time_elapsed = time.time() - cascade_t0
    cascade_score = roc_auc_score(y_test, cascade_preds)

    print("More important features time: %f  Full feature time: %f  Cascade time: %f" %
          (mi_time_elapsed, full_time_elapsed, cascade_time_elapsed))
    print("More important features AUC: %f  Full features AUC: %f  Cascade AUC: %f" %
          (mi_score, full_score, cascade_score))

    # Save top features.
    ft.save_features(top_features, resources_folder + "top_features.dfs")

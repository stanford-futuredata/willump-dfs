import pickle
import time

import featuretools as ft
import pandas as pd
import predict_next_purchase_utils as utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=3,
                             scoring="roc_auc", verbose=False)

    print("All Features AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

    # Select top features.
    clf.fit(X, y)
    top_features = utils.feature_importances(clf, features_encoded, n=20)

    # Train model with top features.
    t0 = time.time()
    top_feature_matrix = ft.calculate_feature_matrix(top_features,
                                                     entityset=es,
                                                     cutoff_time=label_times,
                                                     cutoff_time_in_index=True,
                                                     verbose=False)
    time_elapsed = time.time() - t0
    print("Top Features Calculation Time: %f" % time_elapsed)

    top_feature_matrix = top_feature_matrix.reset_index().merge(label_times)
    top_feature_matrix.drop(["user_id", "time"], axis=1, inplace=True)
    top_feature_matrix = top_feature_matrix.fillna(0)
    top_y = top_feature_matrix.pop("label")

    scores = cross_val_score(estimator=clf, X=top_feature_matrix, y=top_y, cv=3,
                             scoring="roc_auc", verbose=False)

    print("Top Features AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

    clf.fit(top_feature_matrix, top_y)
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

    # print(more_important_features)
    t0 = time.time()
    mi_feature_matrix = ft.calculate_feature_matrix(more_important_features,
                                                    entityset=es,
                                                    cutoff_time=label_times,
                                                    cutoff_time_in_index=True,
                                                    verbose=False)
    time_elapsed = time.time() - t0
    print("More Important Features Calculation Time: %f" % time_elapsed)

    mi_feature_matrix = mi_feature_matrix.reset_index().merge(label_times)
    mi_feature_matrix.drop(["user_id", "time"], axis=1, inplace=True)
    mi_feature_matrix = mi_feature_matrix.fillna(0)
    mi_y = mi_feature_matrix.pop("label")

    # print(less_important_features)
    t0 = time.time()
    li_feature_matrix = ft.calculate_feature_matrix(less_important_features,
                                                    entityset=es,
                                                    cutoff_time=label_times,
                                                    cutoff_time_in_index=True,
                                                    verbose=False)
    time_elapsed = time.time() - t0
    print("Less Important Features Calculation Time: %f" % time_elapsed)

    li_feature_matrix = li_feature_matrix.reset_index().merge(label_times)
    li_feature_matrix.drop(["user_id", "time"], axis=1, inplace=True)
    li_feature_matrix = li_feature_matrix.fillna(0)
    li_y = li_feature_matrix.pop("label")

    scores = cross_val_score(estimator=clf, X=mi_feature_matrix, y=mi_y, cv=3,
                             scoring="roc_auc", verbose=False)

    print("More Important Features AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

    scores = cross_val_score(estimator=clf, X=li_feature_matrix, y=li_y, cv=3,
                             scoring="roc_auc", verbose=False)

    print("Less Important AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

    clf.fit(top_feature_matrix, y)

    # Save model and top features.
    ft.save_features(top_features, resources_folder + "top_features.dfs")
    pickle.dump(clf, open(resources_folder + "model.pk", "wb"))

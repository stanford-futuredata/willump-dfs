import pickle

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import predict_next_purchase_utils as utils
from willump_dfs.evaluation.willump_dfs_graph_builder import *
import pandas as pd
import argparse

resources_folder = "tests/test_resources/predict_next_purchase_resources/"

data_small = "data_small"
data_large = "data_large"
data_full = "data_huge"

data_folder = None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
    parser.add_argument("-d", "--dataset", type=str, help="Cascade threshold")
    args = parser.parse_args()
    cascade_threshold = args.cascades
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

    more_important_features = ft.load_features(resources_folder + "mi_features.dfs")
    full_model = pickle.load(open(resources_folder + "full_model.pk", "rb"))
    less_important_features = ft.load_features(resources_folder + "li_features.dfs")
    small_model = pickle.load(open(resources_folder + "small_model.pk", "rb"))

    _, label_times_test = train_test_split(label_times, test_size=0.2, random_state=42)
    label_times_test = label_times_test.sort_values(by=["user_id"])
    y_test = label_times_test.pop("label")
    print("Test dataset length: %d" % len(label_times_test))

    if cascade_threshold is None:
        print("Without Cascades")
        full_t0 = time.time()
        full_feature_matrix_test = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                               entityset=es,
                                                               cutoff_time=label_times_test)
        full_feature_matrix_test = full_feature_matrix_test.fillna(0)
        full_preds = full_model.predict(full_feature_matrix_test)
        time_elapsed = time.time() - full_t0
        score = roc_auc_score(y_test, full_preds)
    else:
        assert(0.5 <= cascade_threshold <= 1.0)
        print("Cascade Threshold %f" % cascade_threshold)
        cascade_t0 = time.time()
        cascade_preds = willump_dfs_cascade(more_important_features=more_important_features,
                                            less_important_features=less_important_features,
                                            entity_set=es, cutoff_times=label_times_test, small_model=small_model,
                                            full_model=full_model, confidence_threshold=cascade_threshold)
        time_elapsed = time.time() - cascade_t0
        score = roc_auc_score(y_test, cascade_preds)

    print("Time: %f sec  AUC: %f  Throughput: %f rows/sec" % (
        time_elapsed, score, len(label_times_test) / time_elapsed))

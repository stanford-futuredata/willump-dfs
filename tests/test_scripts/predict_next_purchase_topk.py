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
    parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
    parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Cascade threshold")
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "small":
        data_folder = data_small
        valid_size = 10
    elif dataset == "large":
        data_folder = data_large
        valid_size = 1000
    elif dataset == "huge":
        data_folder = data_full
        valid_size = 5000
    else:
        print("Invalid dataset")
        exit(1)
    ratio = pickle.load(open(resources_folder + "cascades_parameters.pk", "rb"))
    top_K = args.top_k

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
    label_times_test = label_times_test.sort_values(by=["user_id"]).iloc[:5000]
    _ = label_times_test.pop("label")
    print("Test dataset length: %d" % len(label_times_test))

    full_t0 = time.time()
    full_feature_matrix_test = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                           entityset=es,
                                                           cutoff_time=label_times_test)
    full_feature_matrix_test = full_feature_matrix_test.fillna(0)
    orig_preds = full_model.predict_proba(full_feature_matrix_test)[:, 1]
    no_cascades_time_elapsed = time.time() - full_t0
    if not args.cascades:
        time_elapsed = no_cascades_time_elapsed
        preds = orig_preds
    else:
        print("Ratio: %f" % ratio)
        cascade_t0 = time.time()
        preds = willump_dfs_topk_cascade(more_important_features=more_important_features,
                                         less_important_features=less_important_features,
                                         entity_set=es, cutoff_times=label_times_test, small_model=small_model,
                                         full_model=full_model, ratio=ratio, top_k=top_K)
        time_elapsed = time.time() - cascade_t0

    orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
    actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
    precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

    orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
    actual_model_sum = sum(preds[actual_model_top_k_idx])

    print("Time: %f sec  Length: %d  Throughput: %f rows/sec" % (
        time_elapsed, len(label_times_test), len(label_times_test) / time_elapsed))
    print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))

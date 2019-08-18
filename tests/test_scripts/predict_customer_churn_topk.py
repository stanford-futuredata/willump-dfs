import argparse
import pickle

import pandas as pd
from featuretools.primitives import make_agg_primitive
from sklearn.metrics import roc_auc_score

from predict_customer_churn_train import partition_to_entity_set, total_previous_month
from willump_dfs.evaluation.willump_dfs_graph_builder import *

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + 'partitions/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
    parser.add_argument("-p", "--partitions", type=int, help="Partitions to use")
    parser.add_argument("-d", "--debug", help="Debug", action="store_true")
    parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
    args = parser.parse_args()
    ratio = pickle.load(open(resources_folder + "cascade_parameters.pk", "rb"))
    top_K = args.top_k

    es, cutoff_times = partition_to_entity_set(args.partitions)
    if args.debug:
        cutoff_times = cutoff_times.sample(n=1000, random_state=42)

    total_previous = make_agg_primitive(total_previous_month, input_types=[ft.variable_types.Numeric,
                                                                           ft.variable_types.Datetime],
                                        return_type=ft.variable_types.Numeric,
                                        uses_calc_time=True)

    more_important_features = ft.load_features(resources_folder + "mi_features.dfs")
    small_model = pickle.load(open(resources_folder + "small_model.pk", "rb"))
    less_important_features = ft.load_features(resources_folder + "li_features.dfs")
    full_model = pickle.load(open(resources_folder + "full_model.pk", "rb"))

    split_date = pd.datetime(2016, 8, 1)
    cutoff_valid = cutoff_times.loc[cutoff_times['cutoff_time'] >= split_date].copy().drop(
        columns=['days_to_churn', 'churn_date'])
    test_y = cutoff_valid.pop("label")

    full_t0 = time.time()
    full_feature_matrix_test = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                           entityset=es,
                                                           cutoff_time=cutoff_valid)
    full_feature_matrix_test = full_feature_matrix_test.replace({np.inf: np.nan, -np.inf: np.nan}). \
            fillna(full_feature_matrix_test.median())
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
                                         entity_set=es, cutoff_times=cutoff_valid, small_model=small_model,
                                         full_model=full_model, ratio=ratio, top_k=top_K)
        time_elapsed = time.time() - cascade_t0

    orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
    actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
    precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

    orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
    actual_model_sum = sum(preds[actual_model_top_k_idx])

    print("Time: %f sec  Length: %d  Throughput: %f rows/sec" % (
        time_elapsed, len(cutoff_valid), len(cutoff_valid) / time_elapsed))
    print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))

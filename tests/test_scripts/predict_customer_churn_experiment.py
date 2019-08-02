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
    parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
    parser.add_argument("-p", "--partitions", type=int, help="Partitions to use")
    parser.add_argument("-d", "--debug", help="Debug", action="store_true")
    args = parser.parse_args()
    cascade_threshold = args.cascades

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

    if cascade_threshold is None:
        print("Without Cascades")
        # Evaluate model.
        sort_col = "pcc_sort_col"
        cutoff_valid[sort_col] = range(len(cutoff_valid))
        mi_t0 = time.time()
        full_feature_matrix = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                          entityset=es,
                                                          cutoff_time=cutoff_valid).sort_values(sort_col).drop(sort_col,
                                                                                                               axis=1)
        mi_feature_matrix_test = full_feature_matrix.replace({np.inf: np.nan, -np.inf: np.nan}). \
            fillna(full_feature_matrix.median())
        mi_preds = full_model.predict(mi_feature_matrix_test)
        time_elapsed = time.time() - mi_t0
        score = roc_auc_score(test_y, mi_preds)
    else:
        assert (0.5 <= cascade_threshold <= 1.0)
        print("Cascade Threshold %f" % cascade_threshold)
        t0 = time.time()
        cascade_preds = willump_dfs_cascade(more_important_features=more_important_features,
                                            less_important_features=less_important_features,
                                            entity_set=es, cutoff_times=cutoff_valid, small_model=small_model,
                                            full_model=full_model, confidence_threshold=cascade_threshold)
        time_elapsed = time.time() - t0
        score = roc_auc_score(test_y, cascade_preds)

    print("Time: %f sec AUC: %f  Throughput: %f rows/sec" % (time_elapsed, score, len(cutoff_valid) / time_elapsed))

import pickle

from featuretools.primitives import make_agg_primitive
from predict_customer_churn_train import partition_to_entity_set, total_previous_month
from sklearn.metrics import roc_auc_score

from willump_dfs.evaluation.willump_dfs_graph_builder import *

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + 'partitions/'

debug = True

if __name__ == '__main__':

    es, cutoff_times = partition_to_entity_set(0)
    if debug:
        cutoff_times = cutoff_times[:1000]

    total_previous = make_agg_primitive(total_previous_month, input_types=[ft.variable_types.Numeric,
                                                                           ft.variable_types.Datetime],
                                        return_type=ft.variable_types.Numeric,
                                        uses_calc_time=True)

    more_important_features = ft.load_features(resources_folder + "mi_features.dfs")
    small_model = pickle.load(open(resources_folder + "small_model.pk", "rb"))
    less_important_features = ft.load_features(resources_folder + "li_features.dfs")
    full_model = pickle.load(open(resources_folder + "full_model.pk", "rb"))

    split_date = pd.datetime(2016, 8, 1)
    cutoff_valid = cutoff_times.loc[cutoff_times['cutoff_time'] >= split_date].copy()
    test_y = cutoff_valid.pop("label")

    # Evaluate model.
    mi_t0 = time.time()
    mi_feature_matrix_test = ft.calculate_feature_matrix(more_important_features,
                                                         entityset=es,
                                                         cutoff_time=cutoff_valid).drop(
        columns=['days_to_churn', 'churn_date'])
    mi_feature_matrix_test = mi_feature_matrix_test.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(mi_feature_matrix_test.median())
    mi_preds = small_model.predict(mi_feature_matrix_test)
    time_elapsed = time.time() - mi_t0
    score = roc_auc_score(test_y, mi_preds)

    print("Time: %f sec AUC: %f  Throughput: %f rows/sec" % (time_elapsed, score, len(cutoff_valid) / time_elapsed))

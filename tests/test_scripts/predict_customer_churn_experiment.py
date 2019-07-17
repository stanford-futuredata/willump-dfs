import pickle
import copy

from featuretools.primitives import make_agg_primitive
from sklearn.metrics import roc_auc_score

from predict_customer_churn_eval import partition_to_entity_set, total_previous_month
from willump_dfs.evaluation.willump_dfs_graph_builder import *

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + 'partitions/'

debug = False

MAX_LOGS_PER_MEMBER = 30


def sample_es(es):
    new_es = copy.deepcopy(es)

    transactions = new_es.entities[1].df
    logs = new_es.entities[2].df

    print("Before: %d transactions, %d logs" % (len(transactions), len(logs)))

    def sample_function(table):
        length = len(table)
        if length > MAX_LOGS_PER_MEMBER:
            table = table.sample(n=MAX_LOGS_PER_MEMBER, random_state=42)
        return table

    logs = logs.groupby("msno", sort=False, as_index=False) \
        .apply(sample_function).set_index("logs_index", drop=False)

    print("After: %d transactions, %d logs" % (len(transactions), len(logs)))

    new_es.entities[1].df = transactions
    new_es.entities[2].df = logs

    return new_es


if __name__ == '__main__':

    es, cutoff_times = partition_to_entity_set(0)
    if debug:
        cutoff_times = cutoff_times[:1000]

    total_previous = make_agg_primitive(total_previous_month, input_types=[ft.variable_types.Numeric,
                                                                           ft.variable_types.Datetime],
                                        return_type=ft.variable_types.Numeric,
                                        uses_calc_time=True)

    use_features = ft.load_features(resources_folder + "mi_features.dfs")
    small_model = pickle.load(open(resources_folder + "small_model.pk", "rb"))

    partitioned_features = willump_dfs_partition_features(use_features)

    # for i, (features) in enumerate(zip(partitioned_features)):
    #     print("%d Features: %s" % (i, features))

    split_date = pd.datetime(2016, 8, 1)
    cutoff_valid = cutoff_times.loc[cutoff_times['cutoff_time'] >= split_date].copy()

    sampled_es = sample_es(es)

    # Evaluate model.
    full_t0 = time.time()
    mi_feature_matrix_test = ft.calculate_feature_matrix(use_features,
                                                         entityset=es,
                                                         cutoff_time=cutoff_valid).drop(
        columns=['days_to_churn', 'churn_date'])
    test_y = mi_feature_matrix_test.pop('label')
    mi_feature_matrix_test = mi_feature_matrix_test.replace({np.inf: np.nan, -np.inf: np.nan}). \
        fillna(mi_feature_matrix_test.median())
    mi_preds = small_model.predict(mi_feature_matrix_test)
    full_time_elapsed = time.time() - full_t0

    mi_score = roc_auc_score(test_y, mi_preds)

    print("Time: %f  AUC: %f  Length %d" % (full_time_elapsed, mi_score, len(cutoff_valid)))

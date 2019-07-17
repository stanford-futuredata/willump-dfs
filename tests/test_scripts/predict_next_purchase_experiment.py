import pickle
import copy

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import predict_next_purchase_utils as utils
from willump_dfs.evaluation.willump_dfs_graph_builder import *
from featuretools.feature_base.feature_base import AggregationFeature, IdentityFeature, TransformFeature

resources_folder = "tests/test_resources/predict_next_purchase_resources/"

data_small = "data_small"
data_large = "data_large"

data_folder = data_large

MAX_ORDERS_PER_PERSON = 5


def sample_es(es):
    new_es = copy.deepcopy(es)

    order_products = new_es.entities[0].df
    orders = new_es.entities[1].df

    print("Before: %d orders, %d order_products" % (len(orders), len(order_products)))

    def sample_function(table):
        length = len(table)
        if length > MAX_ORDERS_PER_PERSON:
            table = table.sample(n=MAX_ORDERS_PER_PERSON, random_state=42)
        return table

    orders = orders.groupby("user_id", sort=False, as_index=False) \
        .apply(sample_function).set_index("order_id", drop=False)

    good_ids = orders["order_id"].values
    order_products = order_products[order_products["order_id"].isin(good_ids)]

    print("After: %d orders, %d order_products" % (len(orders), len(order_products)))

    new_es.entities[0].df = order_products
    new_es.entities[1].df = orders

    return new_es


def agg_depth(feature):
    if isinstance(feature, AggregationFeature):
        ret_val = 1
    else:
        ret_val = 0
    return ret_val + sum(map(agg_depth, feature.base_features))


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

    use_features = ft.load_features(resources_folder + "top_features.dfs")
    model = pickle.load(open(resources_folder + "full_model.pk", "rb"))

    partitioned_features = willump_dfs_partition_features(use_features)

    for i, (features) in enumerate(zip(partitioned_features)):
        print("%d Features: %s" % (i, features))

    label_times_train, label_times_test = train_test_split(label_times, test_size=0.2, random_state=42)
    label_times_train = label_times_train.sort_values(by=["user_id"])
    label_times_test = label_times_test.sort_values(by=["user_id"])
    y_train = label_times_train.pop("label")
    y_test = label_times_test.pop("label")

    sampled_es = sample_es(es)

    sample_partitions = [0]
    no_sample_partitions = [p_id for p_id in range(len(partitioned_features)) if p_id not in sample_partitions]
    sampled_features, no_sampled_features = [], []
    for p_id in sample_partitions:
        sampled_features += partitioned_features[p_id]
    for p_id in no_sample_partitions:
        no_sampled_features += partitioned_features[p_id]

    # Train model with top features.
    full_t0 = time.time()
    if len(sample_partitions) > 0:
        sample_feature_matrix_test = ft.calculate_feature_matrix(sampled_features,
                                                                 entityset=sampled_es,
                                                                 cutoff_time=label_times_test,
                                                                 chunk_size=len(label_times_test))
        sample_feature_matrix_test = sample_feature_matrix_test.fillna(0)
    no_sample_feature_matrix_test = ft.calculate_feature_matrix(no_sampled_features,
                                                                entityset=es,
                                                                cutoff_time=label_times_test,
                                                                chunk_size=len(label_times_test))
    no_sample_feature_matrix_test = no_sample_feature_matrix_test.fillna(0)
    full_time_elapsed = time.time() - full_t0

    feature_names = list(map(lambda x: x.get_name(), use_features))

    if len(sample_partitions) > 0:
        feature_matrix_test = pd.concat([sample_feature_matrix_test, no_sample_feature_matrix_test], axis=1)

        feature_matrix_test = feature_matrix_test[feature_names]
    else:
        feature_matrix_test = no_sample_feature_matrix_test[feature_names]

    y_preds = model.predict(feature_matrix_test)
    score = roc_auc_score(y_test, y_preds)

    print("Time: %f  AUC: %f  Length %d" % (full_time_elapsed, score, len(label_times_test)))

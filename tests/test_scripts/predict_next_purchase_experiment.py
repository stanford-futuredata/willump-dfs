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
    order_products = es.entities[0].df
    orders = es.entities[1].df

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

    es.entity_from_dataframe(entity_id="order_products_sampled",
                             dataframe=order_products,
                             index="order_product_id",
                             variable_types={"aisle_id": ft.variable_types.Categorical,
                                             "reordered": ft.variable_types.Boolean},
                             time_index="order_time")

    es.entity_from_dataframe(entity_id="orders_sampled",
                             dataframe=orders,
                             index="order_id",
                             time_index="order_time")

    es.add_relationship(ft.Relationship(es["orders_sampled"]["order_id"], es["order_products_sampled"]["order_id"]))
    es.add_relationship(ft.Relationship(es["users"]["user_id"], es["orders_sampled"]["user_id"]))


def feature_to_sample(feature, entities_map, new_metadata, new_es):
    if isinstance(feature, IdentityFeature):
        new_variable = copy.copy(feature.variable)
        if new_variable.entity_id in entities_map:
            new_variable.entity = new_es[entities_map[new_variable.entity_id]]
            new_variable.entity_id = entities_map[new_variable.entity_id]
        return IdentityFeature(new_variable)
    elif isinstance(feature, TransformFeature):
        new_base_features = list(map(lambda x: feature_to_sample(x, entities_map, new_metadata, new_es), feature.base_features))
        return TransformFeature(new_base_features, feature.primitive)
    elif isinstance(feature, AggregationFeature):
        new_base_features = list(map(lambda x: feature_to_sample(x, entities_map, new_metadata, new_es), feature.base_features))
        parent_entity_id = feature.parent_entity.id
        if parent_entity_id in entities_map:
            parent_entity_id = entities_map[parent_entity_id]
        parent_entity = new_metadata[parent_entity_id]
        assert(len(feature.relationship_path) == 1)
        direction, relationship = feature.relationship_path[0]
        relationship = copy.copy(relationship)
        relationship.entityset = new_metadata
        if relationship._parent_entity_id in entities_map:
            relationship._parent_entity_id = entities_map[relationship._parent_entity_id]
        if relationship._child_entity_id in entities_map:
            relationship._child_entity_id = entities_map[relationship._child_entity_id]
        return AggregationFeature(base_features=new_base_features, parent_entity=parent_entity,
                                  primitive=feature.primitive, relationship_path=[(direction, relationship)],
                                  use_previous=feature.use_previous, where=feature.where)


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

    # for feature in use_features:
    #     print("Feature: %s  Aggregation Depth: %d" % (str(feature), agg_depth(feature)))

    label_times_train, label_times_test = train_test_split(label_times, test_size=0.2, random_state=42)
    label_times_train = label_times_train.sort_values(by=["user_id"])
    label_times_test = label_times_test.sort_values(by=["user_id"])
    y_train = label_times_train.pop("label")
    y_test = label_times_test.pop("label")

    sample_es(es)
    for i, feature in enumerate(use_features):
        if agg_depth(feature) > 1:
            use_features[i] = feature_to_sample(feature,
                                                {"orders": "orders_sampled",
                                                 "order_products": "order_products_sampled"}, es.metadata, es)
            break

    # Train model with top features.
    full_t0 = time.time()
    top_feature_matrix_test = ft.calculate_feature_matrix(use_features,
                                                          entityset=es,
                                                          cutoff_time=label_times_test,
                                                          chunk_size=len(label_times_test))
    top_feature_matrix_test = top_feature_matrix_test.fillna(0)
    y_preds = model.predict(top_feature_matrix_test)
    full_time_elapsed = time.time() - full_t0

    score = roc_auc_score(y_test, y_preds)

    print("Time: %f  AUC: %f  Length %d" % (full_time_elapsed, score, len(label_times_test)))

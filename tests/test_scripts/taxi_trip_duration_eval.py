import os
import pickle

import featuretools as ft
import numpy as np
import pandas as pd
import taxi_trip_duration_utils as taxi_utils
from featuretools import variable_types as vtypes
from featuretools.primitives import TransformPrimitive
from featuretools.primitives import make_trans_primitive
from featuretools.variable_types import Boolean, LatLong
from sklearn.model_selection import train_test_split
from willump_dfs.evaluation.willump_dfs_graph_builder import *
from willump_dfs.evaluation.willump_dfs_utils import feature_in_list


def haversine(latlong1, latlong2):
    lat_1s = np.array([x[0] for x in latlong1])
    lon_1s = np.array([x[1] for x in latlong1])
    lat_2s = np.array([x[0] for x in latlong2])
    lon_2s = np.array([x[1] for x in latlong2])
    lon1, lat1, lon2, lat2 = map(np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6367 * 2 * np.arcsin(np.sqrt(a))
    return km


def cityblock(latlong1, latlong2):
    lon_dis = haversine(latlong1, latlong2)
    lat_dist = haversine(latlong1, latlong2)
    return lon_dis + lat_dist


def bearing(latlong1, latlong2):
    lat1 = np.array([x[0] for x in latlong1])
    lon1 = np.array([x[1] for x in latlong1])
    lat2 = np.array([x[0] for x in latlong2])
    lon2 = np.array([x[1] for x in latlong2])
    delta_lon = np.radians(lon2 - lon1)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    x = np.cos(lat2) * np.sin(delta_lon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return np.degrees(np.arctan2(x, y))


def is_rush_hour(datetime):
    hour = pd.DatetimeIndex(datetime).hour
    return (hour >= 7) & (hour <= 11)


def is_noon_hour(datetime):
    hour = pd.DatetimeIndex(datetime).hour
    return (hour >= 11) & (hour <= 13)


def is_night_hour(datetime):
    hour = pd.DatetimeIndex(datetime).hour
    return (hour >= 18) & (hour <= 23)


class GeoBox(TransformPrimitive):
    name = "GeoBox"
    input_types = [LatLong]
    return_type = Boolean

    def __init__(self, bottomleft, topright):
        self.bottomleft = bottomleft
        self.topright = topright

    def get_function(self):
        def geobox(latlong, bottomleft=self.bottomleft, topright=self.topright):
            lat = np.array([x[0] for x in latlong])
            lon = np.array([x[1] for x in latlong])
            boxlats = [bottomleft[0], topright[0]]
            boxlongs = [bottomleft[1], topright[1]]
            output = []
            for i, name in enumerate(lat):
                if (min(boxlats) <= lat[i] <= max(boxlats) and
                        min(boxlongs) <= lon[i] <= max(boxlongs)):
                    output.append(True)
                else:
                    output.append(False)
            return output

        return geobox

    def generate_name(self, base_feature_names):
        return u"GEOBOX({}, {}, {})".format(base_feature_names[0],
                                            str(self.bottomleft),
                                            str(self.topright))


resources_folder = "tests/test_resources/predict_taxi_duration/"

TRAIN_DIR = resources_folder + "train.csv"

if __name__ == "__main__":
    data = taxi_utils.read_data(TRAIN_DIR, 100000)
    labels = data[["id", "trip_duration"]]
    labels.set_index("id")
    data = data.drop("trip_duration", axis=1)

    data["pickup_latlong"] = data[['pickup_latitude', 'pickup_longitude']].apply(tuple, axis=1)
    data["dropoff_latlong"] = data[['dropoff_latitude', 'dropoff_longitude']].apply(tuple, axis=1)
    data = data.drop(["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"], axis=1)

    trip_variable_types = {
        'passenger_count': vtypes.Ordinal,
        'vendor_id': vtypes.Categorical,
        'pickup_latlong': LatLong,
        'dropoff_latlong': LatLong,
    }

    es = ft.EntitySet("taxi")

    es.entity_from_dataframe(entity_id="trips",
                             dataframe=data,
                             index="id",
                             time_index='pickup_datetime',
                             variable_types=trip_variable_types)

    es.normalize_entity(base_entity_id="trips",
                        new_entity_id="vendors",
                        index="vendor_id")

    es.normalize_entity(base_entity_id="trips",
                        new_entity_id="passenger_cnt",
                        index="passenger_count")

    cutoff_time = es['trips'].df[['id', 'pickup_datetime']]
    cutoff_time = cutoff_time.merge(labels)

    Bearing = make_trans_primitive(function=bearing,
                                   input_types=[LatLong, LatLong],
                                   commutative=True,
                                   return_type=vtypes.Numeric)

    Cityblock = make_trans_primitive(function=cityblock,
                                     input_types=[LatLong, LatLong],
                                     commutative=True,
                                     return_type=vtypes.Numeric)

    RushHour = make_trans_primitive(function=is_rush_hour,
                                    input_types=[vtypes.Datetime],
                                    return_type=vtypes.Boolean)

    NoonHour = make_trans_primitive(function=is_noon_hour,
                                    input_types=[vtypes.Datetime],
                                    return_type=vtypes.Boolean)

    NightHour = make_trans_primitive(function=is_night_hour,
                                     input_types=[vtypes.Datetime],
                                     return_type=vtypes.Boolean)

    agg_primitives = ['Sum', 'Mean', 'Median', 'Std', 'Count', 'Min', 'Max', 'Num_Unique', 'Skew']
    trans_primitives = [Bearing, Cityblock,
                        GeoBox(bottomleft=(40.62, -73.85), topright=(40.70, -73.75)),
                        GeoBox(bottomleft=(40.70, -73.97), topright=(40.77, -73.9)),
                        RushHour,
                        NoonHour,
                        NightHour,
                        'Day', 'Hour', 'Minute', 'Month', 'Weekday', 'Week', 'Is_weekend']

    # this allows us to create features that are conditioned on a second value before we calculate.
    es.add_interesting_values()

    cutoff_train, cutoff_valid = train_test_split(cutoff_time, test_size=0.2, shuffle=False)
    y_train = cutoff_train.pop("trip_duration")
    y_train = np.log(y_train.values + 1)
    y_valid = cutoff_valid.pop("trip_duration")
    y_valid = np.log(y_valid.values + 1)

    # Calculate feature_matrix using deep feature synthesis
    if not os.path.exists(resources_folder + "top_features.dfs"):
        _, features = ft.dfs(entityset=es,
                             target_entity="trips",
                             trans_primitives=trans_primitives,
                             agg_primitives=agg_primitives,
                             drop_contains=['trips.test_data'],
                             verbose=True,
                             cutoff_time=cutoff_train,
                             approximate='36d',
                             max_depth=4)

    features = ft.load_features(resources_folder + "top_features.dfs")

    feature_matrix_train = ft.calculate_feature_matrix(features,
                                                       entityset=es,
                                                       cutoff_time=cutoff_train,
                                                       approximate='36d')

    feature_matrix_valid = ft.calculate_feature_matrix(features,
                                                       entityset=es,
                                                       cutoff_time=cutoff_valid,
                                                       approximate='36d')

    partitioned_features = willump_dfs_partition_features(features)
    partition_times = willump_dfs_time_partitioned_features(partitioned_features, es, cutoff_train, approximate='36d')
    partition_importances = \
        willump_dfs_mean_decrease_accuracy(features, partitioned_features, feature_matrix_train.values, y_train,
                                           train_function=taxi_utils.train_xgb,
                                           predict_function=taxi_utils.predict_xgb,
                                           scoring_function=taxi_utils.rmse_scoring)

    more_important_features, less_important_features = \
        willump_dfs_find_efficient_features(partitioned_features,
                                            partition_costs=partition_times,
                                            partition_importances=partition_importances)

    for i, (features, cost, importance) in enumerate(zip(partitioned_features, partition_times, partition_importances)):
        print("%d Features: %s\nCost: %f  Importance: %f  Efficient: %r" % (i, features, cost, importance, all(
            feature_in_list(feature, more_important_features) for feature in features)))

    small_model, full_model = willump_dfs_train_models(more_important_features=more_important_features,
                                                       less_important_features=less_important_features,
                                                       entity_set=es,
                                                       training_times=cutoff_train,
                                                       y_train=y_train,
                                                       train_function=taxi_utils.train_xgb,
                                                       approximate='36d')

    mi_t0 = time.time()
    mi_feature_matrix_test = ft.calculate_feature_matrix(more_important_features,
                                                         entityset=es,
                                                         cutoff_time=cutoff_valid,
                                                         approximate='36d')
    mi_preds = taxi_utils.predict_xgb(small_model, mi_feature_matrix_test)
    mi_time_elapsed = time.time() - mi_t0
    mi_score = 1 - taxi_utils.rmse_scoring(y_valid, mi_preds)

    full_t0 = time.time()
    full_feature_matrix_test = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                           entityset=es,
                                                           cutoff_time=cutoff_valid,
                                                           approximate='36d')
    full_preds = taxi_utils.predict_xgb(full_model, full_feature_matrix_test)
    full_time_elapsed = time.time() - full_t0
    full_score = 1 - taxi_utils.rmse_scoring(y_valid, full_preds)

    print("More important features time: %f  Full feature time: %f" %
          (mi_time_elapsed, full_time_elapsed))
    print("More important features RMSE: %f  Full features RMSE: %f" %
          (mi_score, full_score))

    ft.save_features(more_important_features + less_important_features, resources_folder + "top_features.dfs")
    ft.save_features(more_important_features, resources_folder + "mi_features.dfs")
    pickle.dump(small_model, open(resources_folder + "small_model.pk", "wb"))
    pickle.dump(full_model, open(resources_folder + "full_model.pk", "wb"))

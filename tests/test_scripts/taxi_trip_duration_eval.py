import os

import featuretools as ft
import numpy as np
import pandas as pd
import taxi_trip_duration_utils as taxi_utils
from featuretools import variable_types as vtypes
from featuretools.primitives import TransformPrimitive
from featuretools.primitives import make_trans_primitive
from featuretools.variable_types import Boolean, LatLong


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
    data = taxi_utils.read_data(TRAIN_DIR)

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

    # calculate feature_matrix using deep feature synthesis
    if not os.path.exists(resources_folder + "top_features.dfs"):
        _, features = ft.dfs(entityset=es,
                             target_entity="trips",
                             trans_primitives=trans_primitives,
                             agg_primitives=agg_primitives,
                             drop_contains=['trips.test_data'],
                             verbose=True,
                             cutoff_time=cutoff_time,
                             approximate='36d',
                             max_depth=4)
        ft.save_features(features, resources_folder + "top_features.dfs")

    features = ft.load_features(resources_folder + "top_features.dfs")

    feature_matrix = ft.calculate_feature_matrix(features,
                                                 entityset=es,
                                                 cutoff_time=cutoff_time,
                                                 approximate='36d')
    labels = feature_matrix.pop("trip_duration")
    labels = np.log(labels.values + 1)

    model = taxi_utils.train_xgb(feature_matrix, labels)

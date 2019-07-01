from typing import List, Optional

from featuretools.feature_base.feature_base import FeatureBase


def index_feature_in_list(feature: FeatureBase, list: List[FeatureBase]) -> Optional[int]:
    for i, list_feature in enumerate(list):
        if feature.get_name() == list_feature.get_name():
            return i
    return None


def feature_in_list(feature, list):
    return any(feature.get_name() == list_feature.get_name() for list_feature in list)
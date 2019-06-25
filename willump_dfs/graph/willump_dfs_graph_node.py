from typing import List, Optional

from featuretools.feature_base.feature_base import FeatureBase


class WillumpDFSGraphNode(object):

    def __init__(self, feature: FeatureBase, dependencies: Optional[List['WillumpDFSGraphNode']]):
        self._feature = feature
        self._dependencies = dependencies

    def get_feature(self) -> FeatureBase:
        return self._feature

    def get_dependencies(self) -> Optional[List['WillumpDFSGraphNode']]:
        return self._dependencies

    def __repr__(self):
        return self._feature.get_name()

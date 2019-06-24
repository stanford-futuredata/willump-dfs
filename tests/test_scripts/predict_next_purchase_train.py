import featuretools as ft
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import predict_next_purchase_utils as utils

resources_folder = "tests/test_resources/predict_next_purchase_resources/"

data_small = "data_small"
data_large = "data_large"

es = utils.load_entityset(resources_folder + data_small)

label_times = utils.make_labels(es=es,
                                product_name="Banana",
                                cutoff_time=pd.Timestamp('March 15, 2015'),
                                prediction_window=ft.Timedelta("4 weeks"),
                                training_window=ft.Timedelta("60 days"))

feature_matrix, features = ft.dfs(target_entity="users",
                                  cutoff_time=label_times,
                                  training_window=ft.Timedelta("60 days"),  # same as above
                                  entityset=es,
                                  verbose=True)
# encode categorical values
fm_encoded, features_encoded = ft.encode_features(feature_matrix,
                                                  features)

X = utils.merge_features_labels(fm_encoded, label_times)
X.drop(["user_id", "time"], axis=1, inplace=True)
X = X.fillna(0)
y = X.pop("label")

clf = RandomForestClassifier(n_estimators=400, n_jobs=1)
scores = cross_val_score(estimator=clf, X=X, y=y, cv=3,
                         scoring="roc_auc", verbose=True)

print("AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

clf.fit(X, y)
top_features = utils.feature_importances(clf, features_encoded, n=20)

ft.save_features(top_features, resources_folder + "top_features.dfs")

feature_matrix = utils.calculate_feature_matrix(label_times=(label_times, es), features=top_features)
feature_matrix.drop(["user_id", "time"], axis=1, inplace=True)
feature_matrix = feature_matrix.fillna(0)
y = feature_matrix.pop("label")

print(feature_matrix)

clf.fit(feature_matrix, y)
pickle.dump(clf, open(resources_folder + "model.pk", "wb"))
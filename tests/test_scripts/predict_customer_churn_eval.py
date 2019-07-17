import os

import featuretools as ft
import featuretools.variable_types as vtypes
import numpy as np
import pickle
import pandas as pd
from featuretools.primitives import make_agg_primitive
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from willump_dfs.evaluation.willump_dfs_utils import feature_in_list
from willump_dfs.evaluation.willump_dfs_graph_builder import *

N_PARTITIONS = 1000

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + 'partitions/'

debug = False


def pcc_train_function(X, y):
    model = RandomForestClassifier(n_estimators=100, max_depth=40,
                                   min_samples_leaf=50,
                                   n_jobs=1, class_weight='balanced',
                                   random_state=50)
    model.fit(X, y)
    return model


def pcc_eval_function(model, X_valid):
    return model.predict(X_valid)


def partition_to_entity_set(partition, cutoff_time_name='MS-31_labels.csv'):
    """Take in a partition number, create a feature matrix, and save to Amazon S3

    Params
    --------
        partition (int): number of partition
        feature_defs (list of ft features): features to make for the partition
        cutoff_time_name (str): name of cutoff time file
        write: (boolean): whether to write the data to S3. Defaults to True

    Return
    --------
        None: saves the feature matrix to Amazon S3

    """

    partition_dir = partitions_dir + 'p' + str(partition)

    # Read in the data files
    members = pd.read_csv(f'{partition_dir}/members.csv',
                          parse_dates=['registration_init_time'],
                          infer_datetime_format=True,
                          dtype={'gender': 'category'})

    trans = pd.read_csv(f'{partition_dir}/transactions.csv',
                        parse_dates=['transaction_date', 'membership_expire_date'],
                        infer_datetime_format=True)
    logs = pd.read_csv(f'{partition_dir}/logs.csv', parse_dates=['date'])

    # Make sure to drop duplicates
    cutoff_times = pd.read_csv(f'{partition_dir}/{cutoff_time_name}', parse_dates=['cutoff_time'])
    cutoff_times = cutoff_times.drop_duplicates(subset=['msno', 'cutoff_time'])[~cutoff_times['label'].isna()]

    # Create empty entityset
    es = ft.EntitySet(id='customers')

    # Add the members parent table
    es.entity_from_dataframe(entity_id='members', dataframe=members,
                             index='msno', time_index='registration_init_time',
                             variable_types={'city': vtypes.Categorical,
                                             'registered_via': vtypes.Categorical})
    # Create new features in transactions
    trans['price_difference'] = trans['plan_list_price'] - trans['actual_amount_paid']
    trans['planned_daily_price'] = trans['plan_list_price'] / trans['payment_plan_days']
    trans['daily_price'] = trans['actual_amount_paid'] / trans['payment_plan_days']

    # Add the transactions child table
    es.entity_from_dataframe(entity_id='transactions', dataframe=trans,
                             index='transactions_index', make_index=True,
                             time_index='transaction_date',
                             variable_types={'payment_method_id': vtypes.Categorical,
                                             'is_auto_renew': vtypes.Boolean, 'is_cancel': vtypes.Boolean})

    # Add transactions interesting values
    es['transactions']['is_cancel'].interesting_values = [0, 1]
    es['transactions']['is_auto_renew'].interesting_values = [0, 1]

    # Create new features in logs
    logs['total'] = logs[['num_25', 'num_50', 'num_75', 'num_985', 'num_100']].sum(axis=1)
    logs['percent_100'] = logs['num_100'] / logs['total']
    logs['percent_unique'] = logs['num_unq'] / logs['total']
    logs['seconds_per_song'] = logs['total_secs'] / logs['total']

    # Add the logs child table
    es.entity_from_dataframe(entity_id='logs', dataframe=logs,
                             index='logs_index', make_index=True,
                             time_index='date')

    # Add the relationships
    r_member_transactions = ft.Relationship(es['members']['msno'], es['transactions']['msno'])
    r_member_logs = ft.Relationship(es['members']['msno'], es['logs']['msno'])
    es.add_relationships([r_member_transactions, r_member_logs])

    return es, cutoff_times


es, cutoff_times = partition_to_entity_set(0)


def total_previous_month(numeric, datetime, time):
    """Return total of `numeric` column in the month prior to `time`."""
    df = pd.DataFrame({'value': numeric, 'date': datetime})
    previous_month = time.month - 1
    year = time.year

    # Handle January
    if previous_month == 0:
        previous_month = 12
        year = time.year - 1

    # Filter data and sum up total
    df = df[(df['date'].dt.month == previous_month) & (df['date'].dt.year == year)]
    total = df['value'].sum()

    return total


total_previous = make_agg_primitive(total_previous_month, input_types=[ft.variable_types.Numeric,
                                                                       ft.variable_types.Datetime],
                                    return_type=ft.variable_types.Numeric,
                                    uses_calc_time=True)

agg_primitives = ['sum', 'time_since_last', 'avg_time_between', 'all', 'mode', 'num_unique', 'min', 'last',
                  'mean', 'percent_true', 'max', 'std', 'count', total_previous]
trans_primitives = ['is_weekend', 'cum_sum', 'day', 'month', 'diff', 'time_since_previous']
where_primitives = ['sum', 'mean', 'percent_true', 'all', 'any']

if debug:
    cutoff_times = cutoff_times[:1000]

if not os.path.exists(resources_folder + "features.dfs"):
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='members',
                                          cutoff_time=cutoff_times,
                                          agg_primitives=agg_primitives,
                                          trans_primitives=trans_primitives,
                                          where_primitives=where_primitives,
                                          max_depth=1, features_only=False,
                                          verbose=1,
                                          n_jobs=1,
                                          cutoff_time_in_index=False)
    # encode categorical values
    feature_matrix, feature_defs = ft.encode_features(feature_matrix,
                                                      feature_defs)

    # Drop columns with missing values
    missing_pct = feature_matrix.isnull().sum() / len(feature_matrix)
    to_drop = list((missing_pct[missing_pct > 0.9]).index)
    to_drop = [x for x in to_drop if x != 'days_to_churn']

    # Drop highly correlated columns
    threshold = 0.95

    # Calculate correlations
    corr_matrix = feature_matrix.corr().abs()

    # Subset to the upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Identify names of columns with correlation above threshold
    to_drop = to_drop + [column for column in upper.columns if any(
        upper[column] >= threshold)]

    to_drop_index = list(map(lambda x: list(feature_matrix.columns).index(x), to_drop))
    feature_defs = [feature_defs[i] for i in range(len(feature_defs)) if i not in to_drop_index]
    ft.save_features(feature_defs, resources_folder + "features.dfs")

feature_defs = ft.load_features(resources_folder + "features.dfs")

split_date = pd.datetime(2016, 8, 1)
cutoff_train = cutoff_times.loc[cutoff_times['cutoff_time'] < split_date].copy()
cutoff_valid = cutoff_times.loc[cutoff_times['cutoff_time'] >= split_date].copy()
print("%d train rows, %d test rows" % (len(cutoff_train), len(cutoff_valid)))

feature_matrix_train = ft.calculate_feature_matrix(feature_defs,
                                                   entityset=es,
                                                   cutoff_time=cutoff_train,
                                                   verbose=True).drop(columns=['days_to_churn', 'churn_date'])
feature_matrix_train = feature_matrix_train.replace({np.inf: np.nan, -np.inf: np.nan}). \
    fillna(feature_matrix_train.median())

feature_matrix_valid = ft.calculate_feature_matrix(feature_defs,
                                                   entityset=es,
                                                   cutoff_time=cutoff_valid,
                                                   verbose=True).drop(columns=['days_to_churn', 'churn_date'])
feature_matrix_valid = feature_matrix_valid.replace({np.inf: np.nan, -np.inf: np.nan}). \
    fillna(feature_matrix_valid.median())

train_y, test_y = np.array(feature_matrix_train.pop('label')), np.array(feature_matrix_valid.pop('label'))

partitioned_features = willump_dfs_partition_features(feature_defs)

partition_times = willump_dfs_time_partitioned_features(partitioned_features, es, cutoff_train)
partition_importances = \
    willump_dfs_mean_decrease_accuracy(feature_defs, partitioned_features,
                                       feature_matrix_train.values, train_y,
                                       train_function=pcc_train_function,
                                       predict_function=pcc_eval_function,
                                       scoring_function=roc_auc_score)

more_important_features, less_important_features = \
    willump_dfs_find_efficient_features(partitioned_features,
                                        partition_costs=partition_times,
                                        partition_importances=partition_importances)

for i, (features, cost, importance) in enumerate(zip(partitioned_features, partition_times, partition_importances)):
    print("%d Features: %s\nCost: %f  Importance: %f  Efficient: %r" % (i, features, cost, importance, all(
        feature_in_list(feature, more_important_features) for feature in features)))

mi_feature_matrix_train = ft.calculate_feature_matrix(more_important_features,
                                                      entityset=es,
                                                      cutoff_time=cutoff_train,
                                                      verbose=True).drop(
    columns=['days_to_churn', 'churn_date', 'label'])
mi_feature_matrix_train = mi_feature_matrix_train.replace({np.inf: np.nan, -np.inf: np.nan}). \
    fillna(mi_feature_matrix_train.median())
small_model = pcc_train_function(mi_feature_matrix_train, train_y)

full_feature_matrix_train = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                        entityset=es,
                                                        cutoff_time=cutoff_train,
                                                        verbose=True).drop(
    columns=['days_to_churn', 'churn_date', 'label'])
full_feature_matrix_train = full_feature_matrix_train.replace({np.inf: np.nan, -np.inf: np.nan}). \
    fillna(full_feature_matrix_train.median())
full_model = pcc_train_function(full_feature_matrix_train, train_y)

mi_t0 = time.time()
mi_feature_matrix_test = ft.calculate_feature_matrix(more_important_features,
                                                     entityset=es,
                                                     cutoff_time=cutoff_valid).drop(
    columns=['days_to_churn', 'churn_date', "label"])
mi_feature_matrix_test = mi_feature_matrix_test.replace({np.inf: np.nan, -np.inf: np.nan}). \
    fillna(mi_feature_matrix_test.median())
mi_preds = small_model.predict(mi_feature_matrix_test)
mi_time_elapsed = time.time() - mi_t0
mi_score = roc_auc_score(test_y, mi_preds)

full_t0 = time.time()
full_feature_matrix_test = ft.calculate_feature_matrix(more_important_features + less_important_features,
                                                       entityset=es,
                                                       cutoff_time=cutoff_valid).drop(
    columns=['days_to_churn', 'churn_date', "label"])
full_feature_matrix_test = full_feature_matrix_test.replace({np.inf: np.nan, -np.inf: np.nan}). \
    fillna(full_feature_matrix_test.median())
full_preds = full_model.predict(full_feature_matrix_test)
full_time_elapsed = time.time() - full_t0
full_score = roc_auc_score(test_y, full_preds)

print("More important features time: %f  Full feature time: %f" %
      (mi_time_elapsed, full_time_elapsed))
print("More important features AUC: %f  Full features AUC: %f" %
      (mi_score, full_score))

ft.save_features(more_important_features + less_important_features, resources_folder + "top_features.dfs")
ft.save_features(more_important_features, resources_folder + "mi_features.dfs")
pickle.dump(small_model, open(resources_folder + "small_model.pk", "wb"))
pickle.dump(full_model, open(resources_folder + "full_model.pk", "wb"))

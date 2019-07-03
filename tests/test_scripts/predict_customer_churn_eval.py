import pandas as pd
from timeit import default_timer as timer
import numpy as np
from featuretools.primitives import make_trans_primitive

from featuretools.primitives import make_agg_primitive

import featuretools as ft
import featuretools.variable_types as vtypes

N_PARTITIONS = 1000

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + 'partitions/'


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
    cutoff_times = cutoff_times.drop_duplicates(subset=['msno', 'cutoff_time'])

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

start = timer()
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='members',
                                      cutoff_time=cutoff_times,
                                      agg_primitives=agg_primitives,
                                      trans_primitives=trans_primitives,
                                      where_primitives=where_primitives,
                                      max_depth=2, features_only=False,
                                      verbose=1,
                                      n_jobs=1,
                                      cutoff_time_in_index=True)
end = timer()
print(f'{round(end - start)} seconds elapsed.')

print(feature_defs)

print(feature_matrix.head())
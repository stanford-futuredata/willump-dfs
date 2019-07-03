import numpy as np
import pandas as pd
from tqdm import tqdm

N_PARTITIONS = 1000

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + 'partitions/'


def label_customer(customer_id, customer_transactions, prediction_date, churn_days,
                   lead_time=1, prediction_window=1, return_trans=False):
    """
    Make label times for a single customer. Returns a dataframe of labels with times, the binary label,
    and the number of days until the next churn.

    Params
    --------
        customer_id (str): unique id for the customer
        customer_transactions (dataframe): transactions dataframe for the customer
        prediction_date (str): time at which predictions are made. Either "MS" for the first of the month
                               or "SMS" for the first and fifteenth of each month
        churn_days (int): integer number of days without an active membership required for a churn. A churn is
                          defined by exceeding this number of days without an active membership.
        lead_time (int): number of periods in advance to make predictions for. Defaults to 1 (preditions for one offset)
        prediction_window(int): number of periods over which to consider churn. Defaults to 1.
        return_trans (boolean): whether or not to return the transactions for analysis. Defaults to False.

    Return
    --------
        label_times (dataframe): a table of customer id, the cutoff times at the specified frequency, the
                                 label for each cutoff time, the number of days until the next churn for each
                                 cutoff time, and the date on which the churn itself occurred.
        transactions (dataframe): [optional] dataframe of customer transactions if return_trans = True. Useful
                                  for making sure that the function performed as expected

       """

    assert (prediction_date in ['MS', 'SMS']), "Prediction day must be either 'MS' or 'SMS'"
    assert (customer_transactions['msno'].unique() == [customer_id]), "Transactions must be for only customer"

    # Don't modify original
    transactions = customer_transactions.copy()

    # Make sure to sort chronalogically
    transactions.sort_values(['transaction_date', 'membership_expire_date'], inplace=True)

    # Create next transaction date by shifting back one transaction
    transactions['next_transaction_date'] = transactions['transaction_date'].shift(-1)

    # Find number of days between membership expiration and next transaction
    transactions['difference_days'] = (transactions['next_transaction_date'] -
                                       transactions['membership_expire_date']). \
                                          dt.total_seconds() / (3600 * 24)

    # Determine which transactions are associated with a churn
    transactions['churn'] = transactions['difference_days'] > churn_days

    # Find date of each churn
    transactions.loc[transactions['churn'] == True,
                     'churn_date'] = transactions.loc[transactions['churn'] == True,
                                                      'membership_expire_date'] + pd.Timedelta(churn_days + 1, 'd')

    # Range for cutoff times is from first to (last + 1 month) transaction
    first_transaction = transactions['transaction_date'].min()
    last_transaction = transactions['transaction_date'].max()
    start_date = pd.datetime(first_transaction.year, first_transaction.month, 1)

    # Handle December
    if last_transaction.month == 12:
        end_date = pd.datetime(last_transaction.year + 1, 1, 1)
    else:
        end_date = pd.datetime(last_transaction.year, last_transaction.month + 1, 1)

    # Make label times dataframe with cutoff times corresponding to prediction date
    label_times = pd.DataFrame({'cutoff_time': pd.date_range(start_date, end_date, freq=prediction_date),
                                'msno': customer_id
                                })

    # Use the lead time and prediction window parameters to establish the prediction window
    # Prediction window is for each cutoff time
    label_times['prediction_window_start'] = label_times['cutoff_time'].shift(-lead_time)
    label_times['prediction_window_end'] = label_times['cutoff_time'].shift(-(lead_time + prediction_window))

    previous_churn_date = None

    # Iterate through every cutoff time
    for i, row in label_times.iterrows():

        # Default values if unknown
        churn_date = pd.NaT
        label = np.nan
        # Find the window start and end
        window_start = row['prediction_window_start']
        window_end = row['prediction_window_end']
        # Determine if there were any churns during the prediction window
        churns = transactions.loc[(transactions['churn_date'] >= window_start) &
                                  (transactions['churn_date'] < window_end), 'churn_date']

        # Positive label if there was a churn during window
        if not churns.empty:
            label = 1
            churn_date = churns.values[0]

            # Find number of days until next churn by
            # subsetting to cutoff times before current churn and after previous churns
            if not previous_churn_date:
                before_idx = label_times.loc[(label_times['cutoff_time'] <= churn_date)].index
            else:
                before_idx = label_times.loc[(label_times['cutoff_time'] <= churn_date) &
                                             (label_times['cutoff_time'] > previous_churn_date)].index

            # Calculate days to next churn for cutoff times before current churn
            label_times.loc[before_idx, 'days_to_churn'] = (churn_date - label_times.loc[before_idx,
                                                                                         'cutoff_time']). \
                                                               dt.total_seconds() / (3600 * 24)
            previous_churn_date = churn_date
        # No churns, but need to determine if an active member
        else:
            # Find transactions before the end of the window that were not cancelled
            transactions_before = transactions.loc[(transactions['transaction_date'] < window_end) &
                                                   (transactions['is_cancel'] == False)].copy()
            # If the membership expiration date for this membership is after the window start, customer has not churned
            if np.any(transactions_before['membership_expire_date'] >= window_start):
                label = 0

        # Assign values
        label_times.loc[i, 'label'] = label
        label_times.loc[i, 'churn_date'] = churn_date

        # Handle case with no churns
        if not np.any(label_times['label'] == 1):
            label_times['days_to_churn'] = np.nan
            label_times['churn_date'] = pd.NaT

    if return_trans:
        return label_times.drop(columns=['msno']), transactions

    return label_times[['msno', 'cutoff_time', 'label', 'days_to_churn', 'churn_date']].copy()


def make_label_times(transactions, prediction_date, churn_days,
                     lead_time=1, prediction_window=1, ):
    """
    Make labels for an entire series of transactions.

    Params
    --------
        transactions (dataframe): table of customer transactions
        prediction_date (str): time at which predictions are made. Either "MS" for the first of the month
                               or "SMS" for the first and fifteenth of each month
        churn_days (int): integer number of days without an active membership required for a churn. A churn is
                          defined by exceeding this number of days without an active membership.
        lead_time (int): number of periods in advance to make predictions for. Defaults to 1 (preditions for one offset)
        prediction_window(int): number of periods over which to consider churn. Defaults to 1.
    Return
    --------
        label_times (dataframe): a table with customer ids, cutoff times, binary label, regression label,
                                 and date of churn. This table can then be used for feature engineering.
    """

    label_times = []
    transactions = transactions.sort_values(['msno', 'transaction_date'])

    # Iterate through each customer and find labels
    for customer_id, customer_transactions in transactions.groupby('msno'):
        lt_cust = label_customer(customer_id, customer_transactions,
                                 prediction_date, churn_days,
                                 lead_time, prediction_window)

        label_times.append(lt_cust)

    # Concatenate into a single dataframe
    return pd.concat(label_times)


def partition_to_labels(partition_number, prediction_dates=['MS', 'SMS'], churn_periods=[31, 14],
                        lead_times=[1, 1], prediction_windows=[1, 1]):
    """Make labels for all customers in one partition
    Either for one month or twice a month

    Params
    --------
        partition (int): number of partition
        label_type (list of str): either 'MS' for monthly labels or
                                  'SMS' for bimonthly labels
        churn_periods(list of int): number of days with no active membership to be considered a churn
        lead_times (list of int): lead times in number of periods
        prediction_windows (list of int): prediction windows in number of periods

    Returns
    --------
        None: saves the label dataframes with the appropriate name to the partition directory
    """
    partition_dir = partitions_dir + 'p' + str(partition_number)

    # Read in data and filter anomalies
    trans = pd.read_csv(f'{partition_dir}/transactions.csv',
                        parse_dates=['transaction_date', 'membership_expire_date'],
                        infer_datetime_format=True)

    # Deal with data inconsistencies
    rev = trans[(trans['membership_expire_date'] < trans['transaction_date']) |
                ((trans['is_cancel'] == 0) & (trans['membership_expire_date'] == trans['transaction_date']))]
    rev_members = rev['msno'].unique()

    # Remove data errors
    trans = trans.loc[~trans['msno'].isin(rev_members)]

    # Create both sets of lables
    for prediction_date, churn_days, lead_time, prediction_window in zip(prediction_dates, churn_periods, lead_times,
                                                                         prediction_windows):
        cutoff_list = [make_label_times(trans, prediction_date=prediction_date,
                                        churn_days=churn_days, lead_time=lead_time,
                                        prediction_window=prediction_window)]

        # Make label times for all customers
        # Turn into a dataframe
        cutoff_times = pd.concat(cutoff_list)
        cutoff_times = cutoff_times.drop_duplicates(subset=['msno', 'cutoff_time'])

        # Write CSV
        cutoff_times.to_csv(f'{partition_dir}/{prediction_date}-{churn_days}_labels.csv', index=False)


for i in tqdm(range(1000)):
    partition_to_labels(i)

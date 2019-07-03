import pandas as pd
import hashlib
import os
from timeit import default_timer as timer

resources_folder = "tests/test_resources/predict_customer_churn/"
partitions_dir = resources_folder + '/partitions/'
N_PARTITIONS = 1000


def create_blank_partitions():
    """Create blank files in each partition and write the file header"""
    # For each partition create the files with headers
    for i in range(N_PARTITIONS):
        directory = partitions_dir + f'p{i}/'
        # Create five files
        for file in ['transactions.csv', 'train.csv', 'members.csv', 'logs.csv']:
            # Write file header as first line
            with open(directory + file, 'w') as f:
                if file == 'transactions.csv':
                    f.write(','.join(list(transactions.columns)))
                elif file == 'train.csv':
                    f.write(','.join(list(train.columns)))
                elif file == 'members.csv':
                    f.write(','.join(list(members.columns)))
                elif file == 'logs.csv':
                    f.write(','.join(list(logs.columns)))


def id_to_hash(customer_id):
    """Return a 16-bit integer hash of a customer id string"""
    return int(hashlib.md5(customer_id.encode('utf-8')).hexdigest(), 16)


def partition_by_hashing(df, name, progress=None):
    """Partition a dataframe into N_PARTITIONS by hashing the id.

    Params
    --------
        df (pandas dataframe): dataframe for partition. Must have 'msno' column.
        name (str): name of dataframe. Used for saving the row data.
        progress (int, optional): number of rows to be processed before displaying information.
                                  Defaults to None

    Returns:
    --------
        Nothing returned. Dataframe is saved one line at a time as csv files to the N_PARTITIONS
    """
    start = timer()

    # Map the customer id to a partition number
    df['partition'] = df['msno'].apply(id_to_hash) % N_PARTITIONS

    # Iterate through one row at a time
    for partition, grouped in df.groupby('partition'):

        # Don't need to save the partition column
        grouped = grouped.drop(columns='partition')

        # Open file for appending
        with open(partitions_dir + f'p{partition}/{name}.csv', 'a') as f:
            # Write a new line and then data
            f.write('\n')
            grouped.to_csv(f, header=False, index=False)

        # Record progress every `progress` steps
        if progress is not None:
            if partition % progress == 0:
                print(
                    f'{100 * round(partition / N_PARTITIONS, 2)}% complete. {round(timer() - start)} seconds elapsed.',
                    end='\r')

    end = timer()
    if progress is not None:
        print(f'\n{df.shape[0]} rows processed in {round(end - start)} seconds.')


members = pd.read_csv(resources_folder + 'churn/members_v3.csv', nrows=1)
transactions = pd.read_csv(resources_folder + 'churn/transactions.csv', nrows=1)
logs = pd.read_csv(resources_folder + 'churn/user_logs.csv', nrows=1)
train = pd.read_csv(resources_folder + 'churn/train.csv', nrows=1)
if not os.path.exists(partitions_dir + 'p999'):
    # Create a new directory for each partition
    for i in range(N_PARTITIONS):
        os.makedirs(partitions_dir + f'p{i}', exist_ok=False)
create_blank_partitions()

members = pd.read_csv(resources_folder + 'churn/members_v3.csv')
partition_by_hashing(members, name='members', progress=10)
del members
transactions = pd.read_csv(resources_folder + 'churn/transactions.csv')
partition_by_hashing(transactions, name='transactions', progress=10)
del transactions
train = pd.read_csv(resources_folder + 'churn/train.csv')
partition_by_hashing(train, name='train', progress=10)
del train
train = pd.read_csv(resources_folder + 'churn/train_v2.csv')
partition_by_hashing(train, name='train', progress=10)
del train
transactions = pd.read_csv(resources_folder + 'churn/transactions_v2.csv')
partition_by_hashing(transactions, name='transactions', progress=10)
del transactions

chunksize = 1e6
start = timer()

for chunk in pd.read_csv(resources_folder + 'churn/user_logs_v2.csv', chunksize=chunksize):
    partition_by_hashing(chunk, name='logs', progress=None)

    if (i + 1) % 10 == 0:
        print(f'{i * chunksize} rows processed.', end='\r')

end = timer()
print(f'\nuser_logs_v2 Overall time: {round(end - start)} seconds.')

chunksize = 1e7

start = timer()

for i, chunk in enumerate(pd.read_csv(resources_folder + 'churn/user_logs.csv', chunksize=chunksize)):
    partition_by_hashing(chunk, name='logs', progress=None)

    if (i + 1) % 10 == 0:
        print(f'{i * chunksize} rows processed.', end='\r')

end = timer()
print(f'\nuser_logs Overall time: {round(end - start)} seconds.')

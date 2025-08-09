from collections import Counter
from datetime import timedelta, datetime

import pandas
from matplotlib import pyplot as plt
from tqdm import tqdm

from experiments.datasets import AMLSim1mInfo, BerkaPlusInfo, AMLWorldInfo, SAMLDInfo, Dataset


def timegaps(lst):
    differences = [abs(lst[i] - lst[i-1]) for i in range(1, len(lst))]
    return differences

def get_berka_dates():
    start_date = datetime.strptime('930105', '%y%m%d')
    end_date = datetime.strptime('981214', '%y%m%d')

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(int(current_date.strftime('%y%m%d')))
        current_date += timedelta(days=1)
    return date_list

def analysis(accounts, data):
    only_one_sent_per_day = 0
    always_the_same_sent_per_day = 0
    total_sent = 0
    only_one_received_per_day = 0
    total_received = 0
    always_the_same_received_per_day = 0

    for account in tqdm(accounts.sample(10_000)):
        v = data[data['origin_id'] == account]['timestamp'].values
        if len(v) > 0:
            total_sent += 1
            s = set(Counter(v).values())
            if s == {1}:
                only_one_sent_per_day += 1
            if len(s) == 1:
                always_the_same_sent_per_day += 1

    for account in tqdm(accounts.sample(10_000)):
        v = data[data['target_id'] == account]['timestamp'].values
        if len(v) > 0:
            total_received += 1
            s = set(Counter(v).values())
            if s == {1}:
                only_one_received_per_day += 1
            if len(s) == 1:
                always_the_same_received_per_day += 1

    print('sent')
    print('same per day', always_the_same_sent_per_day, always_the_same_sent_per_day / total_sent)
    print('only one', only_one_sent_per_day, only_one_sent_per_day / total_sent)
    print()

    print('received')
    print('same per day', always_the_same_received_per_day, always_the_same_received_per_day / total_received)
    print('only one', only_one_received_per_day, only_one_received_per_day / total_received)

def amlsim():
    data = pandas.read_csv(AMLSim1mInfo.data_dir + 'parsed_transactions.csv', usecols=['timestamp', 'origin_id', 'target_id'])
    accounts = pandas.read_csv(AMLSim1mInfo.data_dir + 'parsed_accounts.csv').sample(frac=1)['account_id']
    analysis(accounts, data)

    for account in tqdm(accounts.sample(10)):
        plt.figure(figsize=(2, 2))
        plt.title(f'account {account}', fontsize=10)
        y, x, _ = plt.hist(data[data['origin_id'] == account]['timestamp'].values, density=False, bins=range(200))
        plt.xticks([0, 99, 199], labels=[1, 100, 200])
        plt.yticks([0, max(y)], labels=[0, round(max(y))])
        plt.ylim(0, max(y) + 1)
        plt.savefig(f'./visualization/timestamps/amlsim/sent_by_{account}.png')
        plt.cla()

    for account in tqdm(accounts.sample(10)):
        plt.figure(figsize=(2, 2))
        plt.title(f'account {account}', fontsize=10)
        y, x, _ = plt.hist(data[data['target_id'] == account]['timestamp'].values, density=False, bins=range(200))
        plt.xticks([0, 99, 199], labels=[1, 100, 200])
        plt.yticks([0, max(y)], labels=[0, round(max(y))])
        plt.ylim(0, max(y) + 1)
        plt.savefig(f'./visualization/timestamps/amlsim/received_by_{account}.png')
        plt.cla()

def berka():
    data = pandas.read_csv(BerkaPlusInfo.data_dir + 'parsed_transactions.csv', usecols=['timestamp', 'origin_id', 'target_id'])

    dates = get_berka_dates()
    data['timestamp'] = data['timestamp'].map({ts: day for ts, day in zip(dates, range(len(dates)))})
    accounts = pandas.read_csv(BerkaPlusInfo.data_dir + 'parsed_accounts.csv')['account_id']

    analysis(accounts, data)

    for account in tqdm(accounts.sample(10)):
        v = data[data['origin_id'] == account]['timestamp'].values
        plt.figure(figsize=(5, 5))
        y, x, _ = plt.hist(v, density=False, bins=len(dates))
        plt.title(f'account {account}', fontsize=10)
        plt.savefig(f'./visualization/timestamps/berka/sent_by_{account}.png')
        plt.cla()

    for account in tqdm(accounts.sample(10)):
        v = data[data['target_id'] == account]['timestamp'].values
        plt.figure(figsize=(5, 5))
        y, x, _ = plt.hist(v, density=False, bins=len(dates))
        plt.title(f'account {account}', fontsize=10)
        plt.savefig(f'./visualization/timestamps/berka/received_by_{account}.png')
        plt.cla()


def amlworld():
    print('loading dataset')
    data = pandas.read_csv(AMLWorldInfo.data_dir + 'parsed_transactions.csv', usecols=['timestamp', 'origin_id', 'target_id'])
    print('parsing timestamps')
    data['timestamp'] = data['timestamp'].apply(lambda ts: ts.split(' ')[0])
    unique_timestamps = data['timestamp'].unique()
    data['timestamp'] = data['timestamp'].map({ts: day for ts, day in zip(unique_timestamps, range(len(unique_timestamps)))})
    print('loading accounts')
    accounts = pandas.read_csv(AMLWorldInfo.data_dir + 'parsed_accounts.csv').sample(frac=1)['account_id']

    only_one_sent_per_day = 0
    always_the_same_sent_per_day = 0
    total_sent = 0
    only_one_received_per_day = 0
    total_received = 0
    always_the_same_received_per_day = 0

    for account in tqdm(accounts):
        v = data[data['origin_id'] == account]['timestamp'].values
        if len(v) > 0:
            total_sent += 1
            s = set(Counter(v).values())
            if s == {1}:
                only_one_sent_per_day += 1
            if len(s) == 1:
                always_the_same_sent_per_day += 1


    print('sent')
    print('same per day', always_the_same_sent_per_day, always_the_same_sent_per_day / total_sent)
    print('only one', only_one_sent_per_day, only_one_sent_per_day / total_sent)
    print()

    for account in tqdm(accounts):
        v = data[data['target_id'] == account]['timestamp'].values
        if len(v) > 0:
            total_received += 1
            s = set(Counter(v).values())
            if s == {1}:
                only_one_received_per_day += 1
            if len(s) == 1:
                always_the_same_received_per_day += 1


    print('received')
    print('same per day', always_the_same_received_per_day, always_the_same_received_per_day / total_received)
    print('only one', only_one_received_per_day, only_one_received_per_day / total_received)


    for account in tqdm(accounts.sample(10)):
        plt.figure(figsize=(2, 2))
        plt.title(f'account {account}', fontsize=10)
        y, x, _ = plt.hist(data[data['origin_id'] == account]['timestamp'].values, density=False, bins=range(len(unique_timestamps)))
        plt.yticks([0, max(y)], labels=[0, round(max(y))])
        plt.ylim(0, max(y) + 1)
        plt.savefig(f'./visualization/timestamps/amlworld/sent_by_{account}.png')
        plt.cla()


def samld():
    data = pandas.read_csv(SAMLDInfo.data_dir + 'parsed_transactions.csv', usecols=['timestamp', 'origin_id', 'target_id'])
    accounts = pandas.read_csv(SAMLDInfo.data_dir + 'parsed_accounts.csv').sample(frac=1)['account_id']
    analysis(accounts, data)
    unique_timestamps = data['timestamp'].unique()
    data['timestamp'] = data['timestamp'].map({ts: day for ts, day in zip(unique_timestamps, range(len(unique_timestamps)))})
    for account in tqdm(accounts.sample(10)):
        plt.figure(figsize=(2, 2))
        plt.title(f'account {account}', fontsize=10)
        y, x, _ = plt.hist(data[data['target_id'] == account]['timestamp'].values, density=False, bins=range(len(unique_timestamps)))
        plt.yticks([0, max(y)], labels=[0, round(max(y))])
        plt.ylim(0, max(y) + 1)
        plt.savefig(f'./visualization/timestamps/samld/sent_by_{account}.png')
        plt.cla()
    for account in tqdm(accounts.sample(10)):
        plt.figure(figsize=(2, 2))
        plt.title(f'account {account}', fontsize=10)
        y, x, _ = plt.hist(data[data['target_id'] == account]['timestamp'].values, density=False, bins=range(len(unique_timestamps)))
        plt.yticks([0, max(y)], labels=[0, round(max(y))])
        plt.ylim(0, max(y) + 1)
        plt.savefig(f'./visualization/timestamps/samld/sent_by_{account}.png')
        plt.cla()



def main():
    amlworld()
    samld()
    berka()
    amlsim()

if __name__ == '__main__':
    main()
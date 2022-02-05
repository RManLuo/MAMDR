import argparse
import csv
import json
import os
import os.path as osp
import random
import sys
from functools import partial
from multiprocessing import Pool

sys.path.append("../../")
import pandas as pd
from tqdm import tqdm

import preprocess_data
from utils import split_stratified_into_train_val_test, shuffle_csv_file

file_abs_path = os.path.split(os.path.realpath(__file__))[0]

HEADER = ['uid', 'pid', 'domain', 'label']


def write_header(domain_save_path):
    train_data_path, val_data_path, test_data_path = [osp.join(domain_save_path, path) for path in
                                                      ['train.csv', 'val.csv', 'test.csv']]
    with open(train_data_path, "w", encoding='utf8', newline='') as train, open(val_data_path, "w", encoding='utf8',
                                                                                newline='') as val, open(
        test_data_path, "w", encoding='utf8', newline='') as test:
        train_writer, val_writer, test_writer = csv.writer(train), csv.writer(val), csv.writer(test)
        [writer.writerow(HEADER) for writer in [train_writer, val_writer, test_writer]]


def file_exist(domain_save_path):
    for path in ['train.csv', 'val.csv', 'test.csv']:
        if not osp.exists(osp.join(domain_save_path, path)):
            return False
    return True


def shuffle_domain_file(domain_save_path, seed):
    for path in ['train.csv', 'val.csv', 'test.csv']:
        filename = osp.join(domain_save_path, path)
        shuffle_csv_file(filename, seed)


def sample_negative_data(subgoup, pid_range, ctr_ratio, domain):
    uid = subgoup[0]
    g = subgoup[1]
    negative_num = int(len(g['pid']) / ctr_ratio)
    exclude_item = g['pid'].unique().tolist()  # Excludes the items clicked by user
    sample_set = [pid for pid in pid_range if pid not in exclude_item]

    negative_data = pd.DataFrame(columns=['uid', 'pid', 'label'])
    try:
        if negative_num > len(sample_set):
            negative_sample_set = sample_set
        else:
            negative_sample_set = random.sample(sample_set, negative_num)

        negative_data['pid'] = negative_sample_set
        negative_data['uid'] = uid
        negative_data['domain'] = domain
        negative_data['label'] = 0
        # negative_train_data = [(uid, negative_pid, 0) for negative_pid in negative_sample_set]
        return negative_data
    except:
        print(exclude_item)
        print(sample_set)
        print(negative_num)
        return negative_data


def save_train_val_test(domain_save_path, df, conf):
    train_data_path, val_data_path, test_data_path = [osp.join(domain_save_path, path) for path in
                                                      ['train.csv', 'val.csv', 'test.csv']]

    with open(train_data_path, "a", encoding='utf8', newline='') as train, open(val_data_path, "a", encoding='utf8',
                                                                                newline='') as val, open(
        test_data_path, "a", encoding='utf8', newline='') as test:
        train_writer, val_writer, test_writer = csv.writer(train), csv.writer(val), csv.writer(test)

        df_train, df_val, df_test = split_stratified_into_train_val_test(df,
                                                                         stratify_colname='label',
                                                                         frac_train=conf['train_val_test'][0],
                                                                         frac_val=conf['train_val_test'][1],
                                                                         frac_test=conf['train_val_test'][2],
                                                                         random_state=conf['seed'])
        train_writer.writerows(df_train[HEADER].values.tolist())
        val_writer.writerows(df_val[HEADER].values.tolist())
        test_writer.writerows(df_test[HEADER].values.tolist())


def split_by_category(processed_file_list, split_save_path, conf):
    n_domain = 0
    for p in processed_file_list:
        (filepath, tempfilename) = os.path.split(p)
        (domain_name, extension) = os.path.splitext(tempfilename)
        print("Process domain: {}...".format(domain_name))

        domain_save_path = osp.join(split_save_path, "domain_{}".format(n_domain))
        # Split if not exists
        if not file_exist(domain_save_path) or conf['rebuild']:
            if not osp.exists(domain_save_path):
                os.makedirs(domain_save_path)

            # Write header
            write_header(domain_save_path)

            if 'random_range' in conf and conf['random_range']:
                ctr_range = conf['ctr_ratio_range']
                ctr_ratio = round(random.uniform(*ctr_range), 2)
                print(f"Ctr ratio: {ctr_ratio}")
            else:
                ctr_ratio = conf['ctr_ratio']

            df = pd.read_csv(p)
            # Drop duplicates
            df.drop_duplicates(inplace=True)

            pid_range = df['pid'].unique().tolist()
            n_uid = len(df['uid'].unique())
            n_pid = len(pid_range)

            df['domain'] = n_domain
            positive_df = df.rename(columns={"uid": "uid", "pid": "pid", "domain": "domain", "score": "label"})
            positive_df['label'] = 1
            save_train_val_test(domain_save_path, positive_df, conf)

            # Group by User
            groups = df.groupby("uid")
            with Pool(conf['cores']) as pool:
                for res in tqdm(
                        pool.imap_unordered(
                            partial(sample_negative_data, pid_range=pid_range, ctr_ratio=ctr_ratio, domain=n_domain),
                            groups), total=len(groups)):
                    if len(res) > 0:
                        save_train_val_test(domain_save_path, res, conf)

            # Save property
            with open(osp.join(domain_save_path, "domain_property.json"), "w") as f:
                json.dump({
                    "domain_name": domain_name,
                    "n_uid": n_uid,
                    "n_pid": n_pid,
                    "ctr_ratio": ctr_ratio,
                    "pid_range": pid_range
                }, f)
        # Shuffle domain file
        shuffle_domain_file(domain_save_path, conf['seed'])
        n_domain += 1

    return n_domain


def split_to_domains(conf):
    raw_data_path = osp.join(file_abs_path, conf['raw_data_path'])
    split_save_path = osp.join(file_abs_path, conf['split_save_path'])
    processed_data_path = osp.join(split_save_path, conf['processed_data_path'])
    categories = conf['categories']
    processed_file_list = preprocess_data.preprocess(categories, processed_data_path, raw_data_path,
                                                     rebuild=conf['rebuild'],
                                                     redownload=conf['redownload'])
    n_domain = 0
    if conf['split_policy'] == "split_by_category":
        n_domain = split_by_category(processed_file_list, split_save_path, conf)

    print("Split {} domains at: {}".format(n_domain, split_save_path))


if __name__ == "__main__":
    # "categories": ["Digital Music", "Movies and TV", "Office Products", "Video Games", "Books"],
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="split config file", default="config.json.example", required=True)
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    random.seed(config['seed'])
    split_to_domains(config)

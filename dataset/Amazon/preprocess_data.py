import gzip
import json
import os
import os.path as osp

import pandas as pd

from get_raw_data import get_raw_data_path
from utils import RawId2Id

file_abs_path = os.path.split(os.path.realpath(__file__))[0]


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    df = []
    for d in parse(path):
        df.append([d['reviewerID'], d['asin'], d['overall']])
    return pd.DataFrame(df, columns=['uid', 'pid', 'score'])


def preprocess(categories, processed_data_path, raw_data_path, rebuild=False, redownload=False):
    uid2id_path = osp.join(processed_data_path, "uid2id.json")
    pid2id_path = osp.join(processed_data_path, "pid2id.json")
    uid2id = RawId2Id(uid2id_path, rebuild)
    pid2id = RawId2Id(pid2id_path, rebuild)
    old_uid2id_hash = hash(uid2id)
    old_pid2id_hash = hash(pid2id)
    processed_file_list = []

    if not osp.exists(processed_data_path):
        os.makedirs(processed_data_path)

    for c in categories:
        f = get_raw_data_path(c, raw_data_path, redownload)
        print("Processing: {}".format(c))
        processed_filename = osp.join(processed_data_path, c.replace(", ", "_").replace(" ", "_") + ".csv")
        if not rebuild and osp.exists(processed_filename):
            print("Category {} exists at {}, Preprocess skip!".format(c, processed_filename))
            processed_file_list.append(processed_filename)
            continue
        df = getDF(f)
        df['uid'] = df['uid'].map(uid2id.fit_transform)
        df['pid'] = df['pid'].apply(pid2id.fit_transform)
        df.to_csv(processed_filename, index=False,
                  columns=['uid', 'pid', 'score'])
        processed_file_list.append(processed_filename)

    print("Category: {}, User: {}, Item: {}".format(len(categories), uid2id.id, pid2id.id))
    # Do not need to save if hash values are the same
    new_uid2id_hash = hash(uid2id)
    new_pid2id_hash = hash(pid2id)
    if new_uid2id_hash != old_uid2id_hash:
        uid2id.export(uid2id_path)
    if new_pid2id_hash != old_pid2id_hash:
        pid2id.export(pid2id_path)

    return processed_file_list


if __name__ == "__main__":
    raw_data_path = osp.join(file_abs_path, "raw_data")
    categories = ["AMAZON FASHION", "Appliances", "All Beauty", "Gift Cards", "Magazine Subscriptions"]
    processed_data_path = osp.join(file_abs_path, "processed_data")
    preprocess(categories, processed_data_path, raw_data_path, rebuild=False)

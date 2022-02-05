import gzip
import json
import os
import os.path as osp

import pandas as pd

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


def preprocess(processed_data_path, raw_data_path, theme_num=-1, rebuild=False):
    uid2id_path = osp.join(processed_data_path, "uid2id.json")
    pid2id_path = osp.join(processed_data_path, "pid2id.json")
    themeid2id_path = osp.join(processed_data_path, "themeid2id.json")
    uid2id = RawId2Id(uid2id_path, rebuild)
    pid2id = RawId2Id(pid2id_path, rebuild)
    themeid2id = RawId2Id(themeid2id_path, rebuild)
    old_uid2id_hash = hash(uid2id)
    old_pid2id_hash = hash(pid2id)
    old_themeid2id_hash = hash(themeid2id)

    df = pd.read_csv(osp.join(raw_data_path, "theme_click_log.csv"))
    df['theme_id'] = df['theme_id'].map(themeid2id.fit_transform)
    # Read user/item embedding
    user_dict = pd.read_csv(osp.join(raw_data_path, "user_embedding.csv"), index_col=0, squeeze=True).to_dict()
    item_dict = pd.read_csv(osp.join(raw_data_path, "item_embedding.csv"), index_col=0, squeeze=True).to_dict()

    # Only keep items and users that has embedding features
    df = df[df['user_id'].isin(user_dict.keys()) & df['item_id'].isin(item_dict.keys())]

    processed_file_list = []

    if not osp.exists(processed_data_path):
        os.makedirs(processed_data_path)

    # Group by Theme
    groups = df.groupby("theme_id")
    for name, group in groups:
        if len(processed_file_list) >= theme_num and theme_num != -1:
            break

        print("Processing: theme {}".format(name))
        processed_filename = osp.join(processed_data_path, "theme_{}.csv".format(name))
        if not rebuild and osp.exists(processed_filename):
            print("Theme {} exists at {}, Preprocess skip!".format(name, processed_filename))
            processed_file_list.append(processed_filename)
            continue
        group['user_id'] = group['user_id'].map(uid2id.fit_transform)
        group['item_id'] = group['item_id'].apply(pid2id.fit_transform)
        group.to_csv(processed_filename, index=False,
                     columns=['user_id', 'item_id'])
        processed_file_list.append(processed_filename)

    if theme_num == -1:
        theme_num = themeid2id.id
    print("Theme: {}, User: {}, Item: {}".format(theme_num, uid2id.id, pid2id.id))
    # Do not need to save if hash values are the same
    new_uid2id_hash = hash(uid2id)
    new_pid2id_hash = hash(pid2id)
    new_themeid2id_hash = hash(themeid2id)
    if new_uid2id_hash != old_uid2id_hash:
        uid2id.export(uid2id_path)
    if new_pid2id_hash != old_pid2id_hash:
        pid2id.export(pid2id_path)
    if new_themeid2id_hash != old_themeid2id_hash:
        themeid2id.export(themeid2id_path)

    # Save user and item embedding dict
    user_emb_dict = {}
    item_emb_dict = {}
    for uid, id in uid2id.raw_id2id.items():
        user_emb_dict[id] = user_dict[uid]
    for pid, id in pid2id.raw_id2id.items():
        item_emb_dict[id] = item_dict[pid]
    with open(osp.join(processed_data_path, "user_emb.json"), 'w') as f:
        json.dump(user_emb_dict, f)

    with open(osp.join(processed_data_path, "item_emb.json"), 'w') as f:
        json.dump(item_emb_dict, f)

    return processed_file_list


if __name__ == "__main__":
    raw_data_path = osp.join(file_abs_path, "raw_data")
    processed_data_path = osp.join(file_abs_path, "processed_data")
    preprocess(processed_data_path, raw_data_path, rebuild=False)

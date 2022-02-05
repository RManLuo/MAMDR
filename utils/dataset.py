import collections
import glob
import json
import math
import os
import os.path as osp

import tensorflow as tf
from tqdm import tqdm


def expand_dim(d, label):
    for key in d:
        d[key] = tf.expand_dims(d[key], axis=-1)
    label = tf.cast(tf.expand_dims(label, axis=-1), tf.float32)
    d['label'] = label  # This only used for customized loss
    return d, label


def get_dataset(file_path, label_name='label', shuffle=True, batch_size=32, shuffle_seed=None,
                shuffle_buffer_size=10000, num_parallel_reads=1):
    file_path = osp.realpath(file_path)
    with os.popen("wc -l {}".format(file_path)) as p:
        n_data = int(p.read().split()[0]) - 1
    steps_per_epoch = int(math.ceil((n_data / float(batch_size))))

    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name=label_name,
        shuffle=shuffle,
        num_epochs=1,
        num_parallel_reads=num_parallel_reads,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_buffer_size=steps_per_epoch,
        shuffle_seed=shuffle_seed)
    dataset = dataset.map(expand_dim, num_parallel_calls=num_parallel_reads)
    return dataset, steps_per_epoch, n_data


class MultiDomainDataset(object):
    def __init__(self, conf):
        self.dataset_path = conf['dataset_path']
        self.domain_split_path = osp.join(self.dataset_path, conf['domain_split_path'])
        self.seed = conf['seed']
        self.conf = conf
        self.batch_size = self.conf['batch_size']
        self.shuffle_buffer_size = self.conf['shuffle_buffer_size']

        with open(osp.join(self.domain_split_path, "processed_data/uid2id.json"), "r") as f:
            raw2id = json.load(f)
            self.n_uid = raw2id['id']
        with open(osp.join(self.domain_split_path, "processed_data/pid2id.json"), "r") as f:
            raw2id = json.load(f)
            self.n_pid = raw2id['id']

        if conf['name'] == "Taobao":
            with open(osp.join(self.domain_split_path, "processed_data/item_emb.json"), "r") as f:
                self.item_emb = json.load(f)
            with open(osp.join(self.domain_split_path, "processed_data/user_emb.json"), "r") as f:
                self.user_emb = json.load(f)

        domains_list = glob.glob(osp.join(self.domain_split_path, "domain_*"))
        domains_list.sort(key=lambda x: int(x.split("_")[-1]))
        self.n_domain = len(domains_list)
        print("Found {} domain, in: {}".format(self.n_domain, self.domain_split_path))

        self.train_dataset = collections.OrderedDict()
        self.val_dataset = collections.OrderedDict()
        self.test_dataset = collections.OrderedDict()
        self.ctr_ratio = collections.OrderedDict()

        for d_path in tqdm(domains_list):
            domain_name = osp.split(d_path)[-1]
            domain_idx = int(domain_name.split("_")[-1])
            # By default train data shuffle at each epoch
            # Fixed data first for splitting into meta-train and meta-val
            shuffle_train = False if "fixed_train" in self.conf and self.conf['fixed_train'] else True
            train_d, n_train_step, n_train = get_dataset(osp.join(d_path, "train.csv"), batch_size=self.batch_size,
                                                         shuffle=shuffle_train,
                                                         num_parallel_reads=self.conf['num_parallel_reads'],
                                                         shuffle_buffer_size=self.shuffle_buffer_size)
            val_d, n_val_step, n_val = get_dataset(osp.join(d_path, "val.csv"), batch_size=self.batch_size,
                                                   shuffle=False,
                                                   shuffle_seed=self.seed,
                                                   num_parallel_reads=self.conf['num_parallel_reads'],
                                                   shuffle_buffer_size=self.shuffle_buffer_size)
            test_d, n_test_step, n_test = get_dataset(osp.join(d_path, "test.csv"), batch_size=self.batch_size,
                                                      shuffle=False,
                                                      shuffle_seed=self.seed,
                                                      num_parallel_reads=self.conf['num_parallel_reads'],
                                                      shuffle_buffer_size=self.shuffle_buffer_size)
            with open(osp.join(osp.join(d_path, "domain_property.json"))) as f:
                domain_property = json.load(f)

            self.ctr_ratio[domain_idx] = domain_property['ctr_ratio']
            self.train_dataset[domain_idx] = {"data": train_d, "n_step": n_train_step, "n_data": n_train}
            self.val_dataset[domain_idx] = {"data": val_d, "n_step": n_val_step, "n_data": n_val}
            self.test_dataset[domain_idx] = {"data": test_d, "n_step": n_test_step, "n_data": n_test}

    def get_train_dataset(self, domain_idx):
        return self.train_dataset[domain_idx]

    def get_val_dataset(self, domain_idx):
        return self.val_dataset[domain_idx]

    def get_test_dataset(self, domain_idx):
        return self.test_dataset[domain_idx]

    @property
    def dataset_info(self):
        total_train, total_val, total_test = 0, 0, 0
        info = {
            'n_user': self.n_uid,
            'n_item': self.n_pid
        }
        for i in self.train_dataset:
            info[i] = {
                "n_train": self.train_dataset[i]['n_data'],
                "n_val": self.val_dataset[i]['n_data'],
                "n_test": self.test_dataset[i]['n_data'],
                "ctr_ratio": self.ctr_ratio[i]
            }
            total_train += self.train_dataset[i]['n_data']
            total_val += self.val_dataset[i]['n_data']
            total_test += self.test_dataset[i]['n_data']
        info["total_train"] = total_train
        info['total_val'] = total_val
        info['total_test'] = total_test
        return info

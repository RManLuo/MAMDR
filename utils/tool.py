import json
import os.path as osp

import pandas
import sklearn
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K


def AUC(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


class SetVarOp(object):
    '''
    Pre-construct the variables update ops.
    '''
    def __init__(self, variables):
        self.assign_ops = []
        self.assign_placeholders = []
        for x in variables:
            tf_dtype = tf.as_dtype(x.dtype.name.split('_')[0])
            if hasattr(x, '_assign_placeholder'):
                assign_op = x._assign_op
                assign_placeholder = x._assign_placeholder
            else:
                assign_placeholder = tf.placeholder(tf_dtype, shape=x.shape)
                assign_op = x.assign(assign_placeholder)
                x._assign_placeholder = assign_placeholder
                x._assign_op = assign_op
            self.assign_ops.append(assign_op)
            self.assign_placeholders.append(assign_placeholder)

    def __call__(self, values):
        '''
        Update variables using values
        :param values: numpy arrays with the same shapes as variables.
        :return:
        '''
        feed_dict = {}
        for holder, value in zip(self.assign_placeholders, values):
            feed_dict[holder] = value
        K.get_session().run(self.assign_ops, feed_dict=feed_dict)


class RawId2Id(object):
    '''
    Map the raw users/items uuid to id and save the map dict into a json file.
    '''
    def __init__(self, path="", rebuild=False):
        self.raw_id2id = {}
        self.id = 0
        if path != "":
            if osp.exists(path) and not rebuild:
                self.load(path)

    def __hash__(self):
        return hash(json.dumps({"id": self.id, "raw_id2id": self.raw_id2id}))

    def fit_transform(self, x):
        '''
        Transform uuid to id
        :param x: string or uuid
        :return: id
        '''
        if x in self.raw_id2id:
            return self.raw_id2id[x]
        else:
            self.raw_id2id[x] = self.id
            self.id += 1
            return self.id - 1

    def export(self, path):
        '''
        Export map dict to file.
        :param path: string
        :return:
        '''
        with open(path, "w") as f:
            json.dump({"id": self.id, "raw_id2id": self.raw_id2id}, f)

    def load(self, path):
        '''
        Load map dict from file
        :param path: string
        :return:
        '''
        with open(path, "r") as f:
            raw2id = json.load(f)
            self.id = raw2id['id']
            self.raw_id2id = raw2id['raw_id2id']


def split_stratified_into_train_val_test(df_input, stratify_colname='label',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    if len(df_temp) > 1:
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                          y_temp,
                                                          stratify=y_temp,
                                                          test_size=relative_frac_test,
                                                          random_state=random_state)
    else:
        df_test = df_temp
        df_val = df_temp.drop(index=df_temp.index)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def shuffle_csv_file(filename, seed=123):
    '''
    Shuffle the csv file by lines.
    :param filename: string, csv filename
    :param seed: int, random seed
    :return:
    '''
    df = pandas.read_csv(filename)
    df_shuffled = sklearn.utils.shuffle(df, random_state=seed)
    df_shuffled.to_csv(filename, index=False)

import argparse
import json

import tensorflow as tf
from tensorflow.python.keras import backend as K

from model_zoo import MAML, Reptile, DomainNegotiation, MLDG
from model_zoo.DeepCTR import DeepCTR
from model_zoo.DeepMTLCTR import DeepMTLCTR
from model_zoo.Star import Star
from model_zoo.mamdr import MAMDR
from model_zoo.uncertainty_weight import UncertaintyWeight
from model_zoo.pcgrad import PCGrad

from utils import MultiDomainDataset


def in_name_list(x, name_list):
    for n in name_list:
        if n in x:
            return True
    return False


def main(config):
    tf.set_random_seed(config['dataset']['seed'])
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    sess = tf.Session(config=c)
    K.set_session(sess)

    # Load Dataset
    dataset = MultiDomainDataset(config['dataset'])

    # Choose Model
    model = None
    deep_ctr_list = ['mlp', 'wdl', 'nfm', 'autoint', 'ccpm', 'pnn', 'deepfm']
    mtl_deep_ctr_list = ['shared_bottom', 'mmoe', 'ple']

    if 'star' in config['model']['name']:
        model = Star(dataset, config)
    elif in_name_list(config['model']['name'], deep_ctr_list):
        model = DeepCTR(dataset, config)
    elif in_name_list(config['model']['name'], mtl_deep_ctr_list):
        model = DeepMTLCTR(dataset, config)
    else:
        print("model: {} not found".format(config['model']['name']))

    if "uncertainty_weight" in config['model']['name']:
        model = UncertaintyWeight(model)

    if "pcgrad" in config['model']['name']:
        model = PCGrad(model)

    if "meta" in config['model']['name']:
        if "domain_negotiation" in config['model']['name']:
            model = DomainNegotiation(model)
        elif "mamdr" in config['model']['name']:
            model = MAMDR(model)
        elif "reptile" in config['model']['name']:
            model = Reptile(model)
        elif "mldg" in config['model']['name']:
            model = MLDG(model)
        else:
            model = MAML(model)

    # Train Model
    if "separate" in config['model']['name']:
        avg_loss, avg_auc, domain_loss, domain_auc = model.separate_train_val_test()
    else:
        model.train()

        print("Test Result: ")
        if "meta" in config['model']['name'] and model.train_config['meta_finetune_step'] > 0:
            graph = tf.get_default_graph()
            if graph.finalized:
                graph._unsafe_unfinalize()

        avg_loss, avg_auc, domain_loss, domain_auc = model.val_and_test("test")

    # Finetune the model on different domains
    if "finetune" in config['model']['name']:
        model.load_model(model.checkpoint_path)
        print("Finetune: ")
        avg_loss, avg_auc, domain_loss, domain_auc = model.separate_train_val_test(init_parms=False)

    model.save_result(avg_loss, avg_auc, domain_loss, domain_auc)

    return avg_loss, avg_auc, domain_loss, domain_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Train config file", required=True)
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)

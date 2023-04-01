# MAMDR: A Model Agnostic Learning Method for Multi-Domain Recommendation
Official code implementation for ICDE 23 paper MAMDR: A Model Agnostic Learning Method for Multi-Domain Recommendation.

Preprint Version: [MAMDR: A Model Agnostic Learning Method for Multi-Domain Recommendation](https://arxiv.org/abs/2202.12524)   
[Slides](./ICDE-23-Slides-MAMDR.pdf)  
## Requirements
```text
tensorflow-gpu==1.12.0
requests==2.26.0
tqdm==4.62.2
pandas==1.1.5
scikit-learn==0.24.2
numpy==1.16.6
deepctr==0.9.0
```
* RTX 2080 + 64G RAM
* python: 3.6
* Ubuntu 20.04
## Data Preprocess

### Amazon dataset
Enter `dataset/Amazon`

raw dataset at：https://nijianmo.github.io/amazon/index.html#complete-data
1. change the split rule in config_*.json.
2. run `split.py --config config_*.json`. It will automatically download and split the domains.

### Taobao dataset
Enter `dataset/Taobao`
1. [download dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=9716)
2. unzip the dataset into `raw_data`
   1. theme_click_log.csv
   2. theme_item_pool.csv
   3. user_item_purchase_log.csv
   4. item_embedding.csv
   5. user_embedding.csv
3. change the dataset config in `config_*.json`. theme_num = -1 denotes using all domains.
4. run `split.py --config config_*.json` to create dataset.

## Preprocessed dataset
You can download the datasets and extract the folders into `dataset/Amazon` and `dataset/Taobao`, respectively.
* [Amazon-6/13](https://drive.google.com/file/d/1JpzaDZioLTlGMJdiLIvrP8m9ngOGe7dI/view?usp=sharing)
* [Taobao-10/20/30](https://drive.google.com/file/d/1L-Y6KZ5-DWKYLspaLo0oxPAQ-MxUgIKe/view?usp=sharing)
## Run experiments

### Run baselines
* change model name in `config.json`
``` python
python3 run.py --config config/Taobao-10/deepctr.json
```
### Run Domain Negotiation
``` python
python3 run.py --config config/Taobao-10/deepctr_DN.json
```
### Rune MAMDR
``` python
python3 run.py --config config/Taobao-10/deepctr_DN+DR.json
```

## Config Description

Model Name：

``basemodel_extenstion(s)``

### Base model：

single domain： `mlp`, `wdl`, `nfm`, `autoint`, `deepfm`

multi tasks： `shared_bottom`, `mmoe`, `ple`

multi domain： `star`

### Extenstion：
```json
default: joint learning
separate: separatly train for each domain
finetune: finetune after joint training
meta: training using MAML
reptile: reptile meta learning
mldg: MLDG
uncertainty_weight: weighted loss
pcgrad: pcgrad        
domain_negotiation: domain negotiation meta learning
mamdr: DN + DR       
batch: using batch for meta learning
```
#### Example
* MLP + Joint:`mlp`
* DeepFM + Joint: `deepfm`
* MLP + Joint + Finetune:`mlp_finetune`
* MLP + MAML:`mlp_meta_finetune`
* MLP + DN: `mlp_meta_domain_negotiation_finetune_`
* MLP + DN + DR : `mlp_meta_mamdr_finetune`

You can find more examples in `config/`.

### Config example
```json
{
  "model": {
    "name": "mlp_meta_mamdr_finetune", // model name
    "norm": "none", // noarmalization method for star: none, bn, pn
    "dense": "dense", // dense for star: dense, star
    "auxiliary_net": false,
    "user_dim": 128,
    "item_dim": 128,
    "domain_dim": 128,
    "auxiliary_dim": 128,
    "hidden_dim": [ // hidden sizes for hidden layers
      256,
      128,
      64
    ],
    "dropout": 0.5
  },
  "train": {
    "load_pretrain_emb": true, // whether load pretrain embding, only support for Taobao
    "emb_trainable": false, // whether train embedding
    "epoch": 99999,
    "learning_rate": 0.001, // inner learning rate
    "meta_learning_rate": 0.1, // outer learning rate
    "domain_meta_learning_rate": 0.1, // Not used
    "merged_method": "plus", 
    "sample_num": 5, // sample number for DR
    "add_query_domain": true,
    "finetune_every_epoch": false,
    "shuffle_sequence": true, // whether shuffle the domain sequence
    "meta_sequence": "random",
    "target_domain": -1, // disabled
    "domain_regulation_step": 0, // disabled
    "meta_train_step": 0, // disabled
    "meta_finetune_step": 0, // disabled
    "meta_split": "train-train", // how to split the query and support set for meta-learning: train-train, meta-train/val, meta-train/val-no-exclusive
    "meta_split_ratio": 0.8,
    "average_meta_grad": "none",
    "meta_parms": [
      "all"
    ],
    "result_save_path": "result",
    "checkpoint_path": "checkpoint",
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "patience": 3,
    "val_every_step": 1,
    "histogram_freq": 0,
    "shuffle_buff_size": 10000
  },
  "dataset": {
    "name": "Taobao",
    "dataset_path": "dataset/Taobao",
    "domain_split_path": "split_by_theme_10",
    "batch_size": 1024,
    "shuffle_buffer_size": 10000,
    "num_parallel_reads": 8,
    "seed": 123
  }
}
```

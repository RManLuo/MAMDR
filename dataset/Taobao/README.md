# Taobao Dataset

1. [download dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=9716)
2. unzip the dataset into `raw_data`
   1. theme_click_log.csv
   2. theme_item_pool.csv
   3. user_item_purchase_log.csv
   4. item_embedding.csv
   5. user_embedding.csv
3. change the dataset config in `config_*.json`. theme_num = -1 denotes using all domains.
4. run `split.py --config config_*.json` to create dataset.
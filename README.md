# Mahjong
## Installation
```bash
conda create -n mahjong python=3.10 -y
conda activate mahjong

pip install PyMahjongGB numpy torch
```

model_origin.py origin model code

model.py the model changed by Peng Yitong

feature_origin.py origin feature extract code

feature.py new feature extract code

pre_train.py and pre_data_process.py is the pre-train code

## Pretrain

put the data.txt into the folder "pretain/data/"

**you can run pre_train.sh to start pretrain**
```bash
./pre_train.sh
```

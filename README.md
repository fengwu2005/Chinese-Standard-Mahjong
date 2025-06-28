# Mahjong
## Installation
```bash
conda create -n mahjong python=3.10 -y
conda activate mahjong
# get your version of torch ready first
pip install -r requirements.txt
```

## Pretrain
download the pretrain data to ./pretrain/data from 
https://disk.pku.edu.cn/link/AAC9250B3B77C7450A9A44EC4653992C66
文件夹名：data
有效期限：2025-07-28 12:01
### preprocess the data
```bash
python pretrain/pre_data_process.py
```
### pretrain
```bash
python pretrain/pre_train.py
```
## RL training
```bash
python scripts/train.py #you can change the config in train.py as you want
```
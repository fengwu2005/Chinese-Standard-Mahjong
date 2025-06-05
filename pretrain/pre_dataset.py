from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False, args = None):
        import json
        if args is not None:
            self.args = args
            self.data_path = args.data
        else:
            self.data_path = 'pretrain/data'
        with open(f'{self.data_path}/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        # print(self.total_matches)
        self.total_samples = sum(self.match_samples)
        # print(self.total_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        # self.cache = {'obs': [], 'mask': [], 'act': []}
        # for i in range(self.matches):
        #     if i % 128 == 0: print('loading', i)
        #     d = np.load('data/%d.npz' % (i + self.begin))
        #     for k in d:
        #         self.cache[k].append(d[k])
            # if i % 128 == 0: print(len(self.cache['obs']))
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        # return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], self.cache['act'][match_id][sample_id]
        # print(index, match_id, sample_id)
        d = np.load(f'{self.data_path}/%d.npz' % (match_id + self.begin))
        # print(d['obs'][0][sample_id])
        return d['obs'][sample_id], d['mask'][sample_id], d['act'][sample_id]
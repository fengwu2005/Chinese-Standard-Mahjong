import torch
import numpy as np
from model import CNNModel
from env.env import MahjongGBEnv
from env.feature import FeatureAgent
class Evaluator:
    def __init__(self, config, model_ckpt=None, baseline_ckpt=None):
        self.device = config.get('device', 'cpu')
        self.eval_episodes = config.get('eval_episodes', 20)
        self.baseline_ckpt = config.get('baseline_ckpt', baseline_ckpt)

        self.env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})

        self.config = config
        # 加载当前模型
        self.model = CNNModel().to(self.device)
        self.baseline_model = CNNModel().to(self.device)

        self.update_model(model_ckpt, baseline_ckpt)

    def update_model(self, model_ckpt=None, baseline_ckpt=None):
        if model_ckpt is not None:
            self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
            self.model.eval()
        if baseline_ckpt is not None:
            self.baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=self.device))
            self.baseline_model.eval()

    def evaluate(self):
        model_scores = []
        baseline_scores = []

        for seat in range(4):
            for ep in range(self.eval_episodes // 4):
                agent_clz_list = [lambda idx: self.ModelAgent(idx, self.baseline_model, self.device)] * 4
                agent_clz_list[seat] = lambda idx: self.ModelAgent(idx, self.model, self.device)
                obs = self.env.reset()
                done = False
                while not done:
                    action_dict = {}
                    for i in range(4):
                        action_dict[f'player_{i+1}'] = self.env.agents[i].act(obs[f'player_{i+1}'])
                    obs, reward, done = self.env.step(action_dict)
                # 统计分数
                model_scores.append(reward[f'player_{seat+1}'])
                for i in range(4):
                    if i != seat:
                        baseline_scores.append(reward[f'player_{i+1}'])

        avg_model = np.mean(model_scores)
        avg_baseline = np.mean(baseline_scores)

        return avg_model, avg_baseline

    class ModelAgent:
        def __init__(self, idx, model, device):
            self.idx = idx
            self.model = model
            self.device = device
        def act(self, obs):
            obs_tensor = torch.tensor(obs['observation']).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(obs['action_mask']).unsqueeze(0).to(self.device)
            logits, _ = self.model({'observation': obs_tensor, 'action_mask': mask_tensor})[:2]
            return torch.argmax(logits, dim=1).item()
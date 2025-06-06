import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.replay_buffer import ReplayBuffer
from scripts.actor import Actor
from scripts.learner import Learner
import time
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 8,
        'episodes_per_actor': 1000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        'ckpt_save_interval': 300,
        'ckpt_save_path': './models/',
        'pretrain_ckpt_path': 'pretrain/ckpt/20250606-140823',
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config["ckpt_save_path"] = os.path.join(config["ckpt_save_path"], timestamp)
    os.makedirs(config["ckpt_save_path"], exist_ok=True)
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()
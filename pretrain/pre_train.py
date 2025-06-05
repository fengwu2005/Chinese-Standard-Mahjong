import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_dataset import MahjongGBDataset, MahjongGBDataset_Allload
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train the model')
    parser.add_argument('--logdir', type=str, default='model/', help='Directory to save the model checkpoints')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save the model checkpoints')
    parser.add_argument('--splitratio', type=float, default=0.9, help='Split ratio for training and validation datasets')
    parser.add_argument('--timestamp', type=str, default=None, help='Number of timesteps to consider in the dataset')
    parser.add_argument('--data', type=str, default='pretrain/data', help='Path to the Mahjong GB data file')
    args = parser.parse_args()

    # trainDataset = MahjongGBDataset(0, args.splitratio, True, args)
    # validateDataset = MahjongGBDataset(args.splitratio, 1, False, args)
    trainDataset = MahjongGBDataset_Allload(0, args.splitratio, True, args)
    validateDataset = MahjongGBDataset_Allload(args.splitratio, 1, False, args)
    loader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = args.batch_size, shuffle = False)
    model = CNNModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    if args.timestamp is not None:
        timestamp = args.timestamp
        args.logdir = os.path.join(args.logdir, timestamp)
        # Find the highest epoch model file in the logdir and load it
        model_files = [f for f in os.listdir(args.logdir) if f.endswith('.pkl')]
        if model_files:
            max_epoch = max([int(f.split('.')[0]) for f in model_files if f.split('.')[0].isdigit()])
            model_path = os.path.join(args.logdir, f"{max_epoch}.pkl")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            print(f"Loaded model from {model_path}")
        else:
            print("No model checkpoint found in the directory.")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.logdir = os.path.join(args.logdir, timestamp)
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)
        
        
    # Train and validate
    for e in tqdm(range(args.epochs), desc='Training', unit='epoch'):
        model.train()
        epoch_loss = 0
        
            # Save the model state_dict
        t3 = time.time()
        for d in tqdm(loader, desc=f'epoch{e}', unit='batch'):
            # 取数据
            # DataLoader 已经自动取数据到 d
            t1 = time.time()
            data_load_time = t1 - t3
            print('Data load time:', data_load_time)

            #input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            #print(input_dict["observation"].shape)
            logits, _ = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()
            compute_time = t3 - t1
            print('Compute time:', compute_time)
        avg_loss = epoch_loss / len(loader)
        writer.add_scalar('Loss/Train', avg_loss, e + 1)

        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            #input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            with torch.no_grad():
                logits, _ = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        writer.add_scalar('Accuracy/Validate', acc, e + 1)
        print('Epoch', e + 1, 'Validate acc:', acc)

        if (e+1) % args.save_interval == 0:
            logdir = args.logdir
            os.makedirs(logdir, exist_ok=True)
            torch.save(model.state_dict(), logdir + '%d.pkl' % e)
            print('Saving model to', logdir + '%d.pkl' % e)
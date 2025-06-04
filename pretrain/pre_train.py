from pre_dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import os
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
    args = parser.parse_args()
    #logdir = 'model/'   
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.logdir = os.path.join(args.logdir, timestamp)
    os.mkdir(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    # Load dataset
    splitRatio = 0.9
    #batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = args.batch_size, shuffle = False)
    
    # Load model
    model = CNNModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    # model.load_state_dict(torch.load("model/checkpoint/6.pkl", map_location = torch.device('cpu')))
    # Train and validate
    for e in tqdm(range(args.epochs), desc='Training', unit='epoch'):
        model.train()
        epoch_loss = 0
        
            # Save the model state_dict
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            # print(input_dict["obs"]["observation"].shape())
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = epoch_loss / len(loader)
        writer.add_scalar('Loss/Train', avg_loss, e + 1)
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
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
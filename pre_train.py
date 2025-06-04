from pre_dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_name", type=str, default="test", help="Experiment name."
    )
    args, _ = parser.parse_known_args()
    args = vars(args)
    exp_name = args['exp_name']

    logdir = 'model/'
    save_dir = os.path.join(logdir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    # os.mkdir(logdir + 'checkpoint')
    
    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # Load model
    model = CNNModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    # model.load_state_dict(torch.load("model/checkpoint/6.pkl", map_location = torch.device('cpu')))
    # Train and validate
    for e in range(16):
        print('Epoch', e)
        torch.save(model.state_dict(), save_dir + '/%d.pkl' % e)
        for i, d in enumerate(loader):
            # input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            # print(input_dict["obs"]["observation"].shape())
            logits, _ = model(input_dict)
            # print(logits.shape, d[2].long().cuda().shape)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 8 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            # input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            input_dict = {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            with torch.no_grad():
                logits, _ = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
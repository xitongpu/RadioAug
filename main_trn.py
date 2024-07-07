import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import dataset2016a, dataset2016b
from models.TRN import TRN
from utils import *


def main():
    
    parser = argparse.ArgumentParser(description='CLDNN for AMC')
    parser.add_argument('--batch-size', type=int, default=500, 
                        help='Input batch size for training (default: 500)')
    parser.add_argument('--test-batch-size', type=int, default=500, 
                        help='Input batch size for testing (default: 500)')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--load', action='store_true', default=True,
                        help='Load the pretrained model for testing')
    parser.add_argument('--dataset', type=str, default='a', 
                        help='Dataset to load: a for RML2016.10A and b for RML2016.10B (default: a)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/TRN_2016a_all.pth', 
                        help='Path of the pre-trained model to be loaded')
    args = parser.parse_args()

    if args.dataset == 'a':
        (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = dataset2016a.load_data()
        n_classes = 11
    else:
        (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = dataset2016b.load_data()
        n_classes = 10

    X_train=np.expand_dims(X_train,axis=1)     # add a channel dimension
    X_test=np.expand_dims(X_test,axis=1)

    loss_func = nn.CrossEntropyLoss()
    
    # Load dataset
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    
    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    net = TRN(img_size=16,
              patch_size=4,
              n_classes=n_classes,
              depth=5,
              n_heads=16,
              embed_dim=256,
              mlp_ratio=1,
              norm_layer=nn.LayerNorm
              ).cuda()
    
    print(net)

    if args.load:
        net.load_state_dict(torch.load(args.checkpoint))
        net_test(net, test_load, 0, loss_func, device)
        return


if __name__ == '__main__':
    main()
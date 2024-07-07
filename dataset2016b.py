import pickle
import numpy as np
from numpy import linalg as la
from scipy.ndimage import zoom
from radioaug import *


def load_data(filename=r'datasets/RML2016.10b.dat'):
    Xd = pickle.load(open(filename,'rb'),encoding='iso-8859-1')  # Xd(120W,2,128) 10calss*20SNR*6000samples
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] # mods['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    X = []
    lbl = []
    train_idx = []
    # val_idx = []
    np.random.seed(2016)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     # ndarray(6000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            train_idx += list(np.random.choice(range(a*6000,(a+1)*6000), size=4800, replace=False))
            # val_idx += list(np.random.choice(list(set(range(a*6000,(a+1)*6000)) - set(train_idx)), size=1200, replace=False))
            a+=1
    X = np.vstack(X)
    n_examples = X.shape[0]
    test_idx = list(set(range(0,n_examples)) - set(train_idx)) #- set(val_idx))
    
    X_train = X[train_idx]
    # X_val = X[val_idx]
    X_test =  X[test_idx]
    
    Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    # Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    ######################################### Translocation
    # Uncomment the following code block when using Translocation.
    # trans_X = []
    # trans_Y = []
    # for i in range(220):  # 11 classes * 20 SNR  
    #     sample_x = X_train[i*800:(i+1)*800].copy()
    #     sample_y = Y_train[i*800:(i+1)*800]
    #     index = np.random.permutation(800)
    #     data_copy = sample_x.copy()[index]

    #     for x,y,lbl_y in zip(sample_x, data_copy, sample_y):
    #         start1 = np.random.randint(int(0.0*128), int(0.2*128))  
    #         start2 = np.random.randint(int(0.2*128), int(0.4*128))  
    #         # segment1 = np.zeros((2, 128), dtype=np.float32)
    #         # segment2 = np.zeros((2, 128), dtype=np.float32)
    #         segment1_1 = x[:, 0:start1]
    #         segment1_2 = y[:, start2:]
    #         segment2_1 = y[:, 0:start2]
    #         segment2_2 = x[:, start1:]
    #         segment1 = np.concatenate((segment1_1, segment1_2), 1)
    #         segment2 = np.concatenate((segment2_1, segment2_2), 1)

    #         if segment1.shape[1] < 128:
    #             segment1_0 = np.zeros((2, 128), dtype=np.float32)
    #             insert_1 = np.random.randint(0, 128 - segment1.shape[1] + 1)
    #             segment1_0[:, insert_1:insert_1 + segment1.shape[1]] = segment1
    #             segment1 = segment1_0
    #         elif segment1.shape[1] > 128:
    #             insert_1 = np.random.randint(0, segment1.shape[1] - 128 + 1)
    #             segment1 = segment1[:, insert_1:insert_1 + 128]

    #         if segment2.shape[1] < 128:
    #             segment2_0 = np.zeros((2, 128), dtype=np.float32)
    #             insert_2 = np.random.randint(0, 128 - segment2.shape[1] + 1)
    #             segment2_0[:, insert_2:insert_2 + segment2.shape[1]] = segment2
    #             segment2 = segment2_0
    #         elif segment2.shape[1] > 128:
    #             insert_2 = np.random.randint(0, segment2.shape[1] - 128 + 1)
    #             segment2 = segment2[:, insert_2:insert_2 + 128]

    #         trans_X.append(np.expand_dims(segment1, 0))
    #         trans_X.append(np.expand_dims(segment2, 0))
    #         trans_Y.append(lbl_y)
    #         trans_Y.append(lbl_y)

    # Uncomment the following two lines of code only when Translocation is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(trans_X)], 0)
    # Y_train = np.concatenate([Y_train, np.array(trans_Y)], 0)

    ######################################### Ring 
    # Uncomment the following code block when using Ring.
    # ring_X = []
    # ring_Y = []
    # for i, signal in enumerate(X_train):
    #     signal = Ring(signal)
    #     ring_X.append(np.expand_dims(signal, 0))
    #     ring_Y.append(Y_train[i])

    # Uncomment the following two lines of code only when Ring is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(ring_X)], 0)
    # Y_train = np.concatenate([Y_train, ring_Y], 0)

    ######################################### Breakage
    # Uncomment the following code block when using Breakage.
    # brkg_X = []
    # brkg_Y = []
    # for i, signal in enumerate(X_train):
    #     signal1, signal2 = Breakage(signal)
    #     brkg_X.append(np.expand_dims(signal1,0))
    #     brkg_X.append(np.expand_dims(signal2,0))
    #     brkg_Y.append(Y_train[i])
    #     brkg_Y.append(Y_train[i])

    # Uncomment the following two lines of code only when Breakage is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(brkg_X)], 0)
    # Y_train = np.concatenate([Y_train, brkg_Y], 0)

    ######################################### Inversion
    # Uncomment the following code block when using Inversion.
    # inv_X = []
    # inv_Y = []
    # for i, signal in enumerate(X_train):
    #     signal = Inversion(signal)
    #     inv_X.append(np.expand_dims(signal,0))
    #     inv_Y.append(Y_train[i])

    # Uncomment the following two lines of code only when Inversion is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(inv_X)], 0)
    # Y_train = np.concatenate([Y_train, inv_Y], 0)

    ######################################### Terminal Deletion
    # Uncomment the following code block when using Terminal Deletion.
    # termdel_X = []
    # termdel_Y = []
    # for i, sample in enumerate(X_train):
    #     sample = Termdel(np.expand_dims(sample,0), (2,128), (0, 25))
    #     termdel_X.append(sample)
    #     termdel_Y.append(Y_train[i])

    # Uncomment the following two lines of code only when Terminal Deletion is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(termdel_X)], 0)
    # Y_train = np.concatenate([Y_train, termdel_Y], 0)

    ######################################### Interstitial Deletion
    # Uncomment the following code block when using Interstitial Deletion.
    # intdel_X = []
    # intdel_Y = []
    # for i, signal in enumerate(X_train):
    #     signal1, signal2 = cutmid_and_catend(signal)
    #     intdel_X.append(np.expand_dims(signal1,0))
    #     intdel_X.append(np.expand_dims(signal2,0))
    #     intdel_Y.append(Y_train[i])
    #     intdel_Y.append(Y_train[i])

    # Uncomment the following two lines of code only when Interstitial Deletion is applied alone.
    # X_train = np.concatenate([X_train, np.vstack(intdel_X)], 0)
    # Y_train = np.concatenate([Y_train, intdel_Y], 0)

    ########################################## 

    # Uncomment the following two lines only when all six augmentation methods are used at the same time.
    # X_train = np.concatenate([X_train, np.vstack(trans_X), np.vstack(ring_X), np.vstack(brkg_X), np.vstack(inv_X), np.vstack(termdel_X), np.vstack(intdel_X)], 0) 
    # Y_train = np.concatenate([Y_train, trans_Y, ring_Y, brkg_Y, inv_Y, termdel_Y, intdel_Y], 0) 

    return (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx, test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train,Y_train), (X_test,Y_test), (train_idx,test_idx) = load_data()

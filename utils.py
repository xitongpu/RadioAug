import torch

def net_test(net, test_data_load, epoch, loss_func, device):
    net.eval()

    ok = 0
    val_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_data_load):
            signal, label = data
            signal, label = signal.to(device), label.to(device)

            outs = net(signal)
            loss = loss_func(outs, label.long())
            val_loss += loss.item()
            _, pre = torch.max(outs.data, 1)
            ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_load.dataset))
    loss_mean = val_loss / (i + 1)

    print('EPOCH:{}, LOSS:{}, ACC:{}\n'.format(epoch, loss_mean, acc))

    return loss_mean
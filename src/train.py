import torch
from torch.utils.data import DataLoader
from src.model.neural_net import CNN
from src.backward import backward


def train(model: CNN, data_loader: DataLoader, epochs: int,
          device: str, lr: float, t_max: int,
          lr_min: float):
    # model - neural net model to train
    # data_loader - data loader with pin_memory=device
    # epochs - the num of epochs
    # device - cpu or cuda
    # lr - learning rate
    # t_max - lr decreasing epochs (for lr scheduler)
    # lr_min - lr at the end of decreasing (lr scheduler)

    if device == "cuda" and not torch.cuda.is_available():
        return "Cuda is not available"

    model = model.to(device)
    model.train()  # enable train mode

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr_min)

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for i_batch, batch in enumerate(data_loader):
            batch['labels'] = batch['labels'].to(device)
            batch['image'] = batch['image'].to(device)

            logits = model(batch['image'])
            backward(logits, batch['labels'], optim)

        scheduler.step()

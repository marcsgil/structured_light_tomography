import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import models
import training
from torchvision.transforms import v2
import h5py
from torch.utils.data import random_split

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
BATCH_SIZE = 64


def format_data(x):
    mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
    std = [x[:, n, :, :].std() for n in range(x.shape[1])]

    X = v2.Compose([
            torch.from_numpy,
            v2.Normalize(mean=mean, std=std),
            v2.Resize((64, 64)),
        ])(x).to(device)
    return X

for order in range(1,2):
    model = models.LeNet(64,64,2, (order + 1)**2 - 1).to(device)

    with h5py.File('../Data/Training/mixed_intense.h5', 'r') as f:
            images = format_data(f[f'images_order{order}'][:])
            labels = torch.from_numpy(f[f'labels_order{order}'][:])

    dset = TensorDataset(images, labels)

    train_size = int(0.85 * len(dset))
    test_size = len(dset) - train_size

    train_dataset, test_dataset = random_split(dset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    save_path = f"TrainedModels"

    writer = SummaryWriter(save_path)
    early_stopping = training.EarlyStopping(patience=40,save_path=save_path)
    for t in range(200):
        epoch = t+1
        print(f"-------------------------------\nEpoch {epoch}")
        training.train(model, train_loader, loss_fn, optimizer, device)
        val_loss = training.test(model, test_loader, loss_fn, device, epoch, writer, verbose=True)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("Done!")
    writer.close()
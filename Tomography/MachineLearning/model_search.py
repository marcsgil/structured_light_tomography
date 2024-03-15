import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import models
import training
from torchvision.transforms import v2
import torch.nn.functional as F
import h5py
from torch.utils.data import random_split

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

order = 4

MODEL_NAME = "EfficientNetB0"
#model = models.DefaultConvNet(64,64,2, (order+1)**2-1).to(device)
#model = models.MobileNet((order+1)**2-1).to(device)
#model = models.ResNet9(2, (order+1)**2-1).to(device)
#model = models.ResNet18((order+1)**2-1).to(device)
#model = models.ResNet34((order+1)**2-1).to(device)
model = models.EfficientNetB0((order+1)**2-1).to(device)

BATCH_SIZE = 2^10

def format_data(x):
    mean = [x[:, n, :, :].mean() for n in range(x.shape[1])]
    std = [x[:, n, :, :].std() for n in range(x.shape[1])]

    X = v2.Compose([
            torch.from_numpy,
            v2.Normalize(mean=mean, std=std),
            v2.Resize((64, 64)),
        ])(x).to(device)
    return X

with h5py.File('Data/Training/mixed_intense.h5', 'r') as f:
        images = format_data(f[f'images_order{order}'][:])
        labels = torch.from_numpy(f[f'labels_order{order}'][:]).to(device)

dset = TensorDataset(images, labels)

train_size = int(0.85 * len(dset))
test_size = len(dset) - train_size

train_dataset, test_dataset = random_split(dset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=.5, min_lr = 1e-5)

save_path = f"Results/MachineLearningModels/Intense/ModelSeach/{MODEL_NAME}/Mixed_Order{order}"

writer = SummaryWriter(save_path)
early_stopping = training.EarlyStopping(patience=40,save_path=save_path,delta=0.00001)
for t in range(300):
    epoch = t+1
    print(f"-------------------------------\nEpoch {epoch}")
    training.train(model, train_loader, loss_fn, optimizer, device)
    val_loss = training.test(model, test_loader, loss_fn, device, epoch, writer, verbose=True)
    scheduler.step(val_loss)
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break
print("Done!")
writer.close()
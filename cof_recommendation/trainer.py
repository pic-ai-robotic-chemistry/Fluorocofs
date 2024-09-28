import os
import torch.nn
from torch import optim
from torch.optim import lr_scheduler
from data_preprocess import pretraining_data_preprocess
from torch.utils.data import TensorDataset, random_split

from siamese_model import *


def generate_training_dataset_from_dataframe(dataset):
    label = dataset['rating'].numpy()
    dataset = dataset.drop(columns=['rating'])
    x1 = []
    x2 = []
    target = []
    for i in range(len(dataset)):
        for j in range(i+1, dataset):
            x1.append(dataset.iloc[i].numpy())
            x2.append(dataset.iloc[j].numpy())
            if label[i] == label[j]:
                target.append(1.)
            else:
                target.append(0.)
    return torch.Tensor(x1), torch.Tensor(x2), torch.Tensor(target)


def generate_training_dataset_from_numpy(vector, label):
    x1 = []
    x2 = []
    target = []
    for i in range(len(vector)):
        for j in range(i + 1, len(vector)):
            x1.append(vector[i])
            x2.append(vector[j])
            if label[i] == label[j]:
                target.append(1.)
            else:
                target.append(0.)
    return torch.Tensor(x1), torch.Tensor(x2), torch.Tensor(target)


def training(model, vector=None, label=None, dataset=None, n_epochs=20):
    lr = 1e-3
    margin = 1.
    loss_fn = ContrastiveLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    if vector is not None:
        x1, x2, target = generate_training_dataset_from_numpy(vector, label)

    else:
        x1, x2, target = generate_training_dataset_from_dataframe(dataset)

    model.train()
    losses = []
    total_loss = 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        outputs1, outputs2 = model(x1, x2)
        loss_outputs = loss_fn(outputs1, outputs2, target)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        message = 'Epoch: {}/{}. Train set: Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item())
        print(message)


def pretraining(validation=0.2):
    x, train_y1, train_y2 = pretraining_data_preprocess()
    y1 = torch.tensor(train_y1.to_numpy(), dtype=torch.float)
    y2 = torch.tensor(train_y2.to_numpy(), dtype=torch.float)
    train_y = torch.stack([y1, y2], axis=1)
    train_x = torch.tensor(x.to_numpy(), dtype=torch.float)

    embed_model = EmbeddingNet()
    output_model = PredictionNet()
    model = PretrainSiameseNet(embed_model, output_model)

    optimizer = optim.Adam(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    loss_fn = torch.nn.MSELoss()

    model.train()
    n_epochs = 20

    if validation is None:
        for epoch in range(n_epochs):
            output = model(train_x)
            loss = loss_fn(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            message = 'Epoch: {}/{}. Train set: Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item())
            print(message)
        model.save_pretrain_state_dict()

    else:
        # random split for validation
        dataset = TensorDataset(train_x, train_y)
        train_size = int((1-validation)*len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_x, train_y = zip(*train_dataset)
        test_x, test_y = zip(*test_dataset)
        train_x = torch.stack(train_x)
        train_y = torch.stack(train_y)
        test_x = torch.stack(test_x)
        test_y = torch.stack(test_y)

        for epoch in range(n_epochs):
            output = model(train_x)
            loss = loss_fn(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            val_loss = loss_fn(model(test_x), test_y)

            message = 'Epoch: {}/{}. Train set: Loss: {:.4f}, Test set: Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item(), val_loss.item())
            print(message)


def init_pretrain_siamese_model(name='pretrain_parameters.pth'):
    if not os.path.exists(name):
        pretraining(validation=None)

    pretrain_model = EmbeddingNet()
    embed_model = EmbeddingNet2()
    model = PretrainSiameseNet(pretrain_model, embed_model)
    model.load_pretrain_state_dict()
    model.freeze_pretrain()
    return model


def train_pretrain_siamese_model(model, vector=None, label=None, dataset=None, n_epochs=20):

    lr = 1e-3
    margin = 1.
    loss_fn = ContrastiveLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    if vector is not None:
        x1, x2, target = generate_training_dataset_from_numpy(vector, label)

    else:
        x1, x2, target = generate_training_dataset_from_dataframe(dataset)

    model.train()
    losses = []
    total_loss = 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs1 = model(x1)
        outputs2 = model(x2)
        loss_outputs = loss_fn(outputs1, outputs2, target)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        message = 'Epoch: {}/{}. Train set: Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item())
        print(message)

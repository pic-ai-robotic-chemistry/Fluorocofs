import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor


DEVICE = 'CPU'


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet2(nn.Module):
    def __init__(self):
        super(EmbeddingNet2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class PretrainSiameseNet(nn.Module):

    def __init__(self, pretrain_net, embedding_net):
        super().__init__()
        self.pretrain_net = pretrain_net
        self.embedding_net = embedding_net

    def forward(self, x):
        return self.embedding_net(self.pretrain_net(x))

    def freeze_pretrain(self):
        for param in self.pretrain_net.parameters():
            param.requires_grad = False

    def save_pretrain_state_dict(self, name='pretrain_parameters.pth'):
        torch.save(self.pretrain_net.state_dict(), name)

    def load_pretrain_state_dict(self, name='pretrain_parameters.pth'):
        self.pretrain_net.load_state_dict(torch.load(name))


class PredictionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

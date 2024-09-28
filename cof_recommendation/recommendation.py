import torch

from siamese_model import SiameseNet, EmbeddingNet, EmbeddingNet2, PretrainSiameseNet
from data_preprocess import get_vector
from trainer import training, train_pretrain_siamese_model, init_pretrain_siamese_model
import numpy as np
import random


# All possible aldehyde and amine pairs. Please change the list according to your dataset.
POSSIBLE_ALD = ['ald_1', 'ald_5', 'ald_7', 'ald_8', 'ald_10', 'ald_11', 'ald_13', 'ald_20', 'ald_21', 'ald_23', 'ald_24', 'ald_25']
POSSIBLE_AMINE = ['amine_8', 'amine_10', 'amine_11', 'amine_12', 'amine_15', 'amine_16', 'amine_17']


class CofRecommendation:
    # This class is used to recommend the next pair of COFs to evaluate based on the trained model

    def __init__(self, ald=None, amine=None, pretrain=True):

        if amine is None:
            amine = POSSIBLE_AMINE
        if ald is None:
            ald = POSSIBLE_ALD
        self.pretrain = pretrain

        if self.pretrain:
            pretrain_model = init_pretrain_siamese_model()
            embedding_model = EmbeddingNet2()
            self.model = PretrainSiameseNet(pretrain_model, embedding_model)
        else:
            embedding_model = EmbeddingNet()
            self.model = SiameseNet(embedding_model)

        self.chosen_points = []
        self.vector = []
        self.evaluations = []

        # Aldehyde and amine available
        self.ald = ald
        self.amine = amine

        # All possible pairs combination
        self.all_pairs = []
        for ald in self.ald:
            for amine in self.amine:
                self.all_pairs.append((ald, amine))

    def random_init(self, n_sample=5, seed=42):
        # Randomly select n_sample pairs from all possible pairs
        random.seed(seed)
        random.shuffle(self.all_pairs)
        return self.all_pairs[:n_sample]

    def register(self, pair, evaluation):
        # register the pair and its evaluation score
        # pair: (ald, amine)
        self.chosen_points.append(pair)
        self.vector.append(get_vector(*pair))
        self.evaluations.append(evaluation)

    def evaluated(self, pair):
        # check if the pair has been evaluated
        if pair in self.chosen_points:
            return True
        return False

    def train_model(self, epochs=20):
        # train the Siamese model with the registered data
        if self.pretrain:
            train_pretrain_siamese_model(self.model, self.vector, self.evaluations, n_epochs=epochs)
        else:
            training(self.model, self.vector, self.evaluations, n_epochs=epochs)

    def suggest(self):
        # suggest the next pair to evaluate based on the trained model
        self.train_model()
        self.model.eval()
        x1 = []
        x2 = []
        x1_count = []
        pair_name = []
        label2 = []
        for pair in self.all_pairs:
            count = 0
            if self.evaluated(pair):
                continue
            pair_vector = get_vector(pair[0], pair[1])
            for i, evaluated_pair in enumerate(self.chosen_points):
                evaluated_vector = get_vector(evaluated_pair[0], evaluated_pair[1])
                x1.append(pair_vector)
                x2.append(evaluated_vector)
                label2.append(self.evaluations[i])
                count += 1
            x1_count.append(count)
            pair_name.append(pair)

        x1 = torch.Tensor(np.array(x1))
        x2 = torch.Tensor(np.array(x2))
        label2 = torch.Tensor(np.array(label2))
        output1, output2 = self.model(x1, x2)
        weighted_distances = label2 / torch.exp((output2 - output1).pow(2).sum(1))

        # get average score
        pointer = 0
        max_score = -1000000
        chosen_pair = None
        for i, count in enumerate(x1_count):
            score = weighted_distances[pointer:pointer + count].sum() / count
            pointer = pointer + count
            if score > max_score:
                max_score = score
                chosen_pair = pair_name[i]
        return chosen_pair

    def suggest_batch(self, batch_size=5):
        # suggest the next batch of pairs to evaluate based on the trained model
        self.train_model()
        self.model.eval()
        x1 = []
        x2 = []
        x1_count = []
        x1_score = []
        pair_name = []
        label2 = []
        for pair in self.all_pairs:
            count = 0
            if self.evaluated(pair):
                continue
            pair_vector = get_vector(pair[0], pair[1])
            for i, evaluated_pair in enumerate(self.chosen_points):
                evaluated_vector = get_vector(evaluated_pair[0], evaluated_pair[1])
                x1.append(pair_vector)
                x2.append(evaluated_vector)
                label2.append(self.evaluations[i])
                count += 1
            x1_count.append(count)
            pair_name.append(pair)

        x1 = torch.Tensor(np.array(x1))
        x2 = torch.Tensor(np.array(x2))
        label2 = torch.Tensor(np.array(label2))
        if self.pretrain:
            output1 = self.model(x1)
            output2 = self.model(x2)
        else:
            output1, output2 = self.model(x1, x2)
        weighted_distances = label2 / torch.exp((output2 - output1).pow(2).sum(1))

        # get average score
        pointer = 0
        for i, count in enumerate(x1_count):
            score = weighted_distances[pointer:pointer + count].sum() / count
            x1_score.append(score.detach().numpy())
            pointer = pointer + count
        x1_score = np.array(x1_score)
        chosen_index = np.argsort(x1_score)[-batch_size:]
        chosen_pair = []
        for index in chosen_index:
            chosen_pair.append(pair_name[index])
        return chosen_pair, x1_score[-batch_size:]

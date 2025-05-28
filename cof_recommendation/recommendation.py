import torch

from siamese_model import SiameseNet, EmbeddingNet, EmbeddingNet2, PretrainSiameseNet
from data_preprocess import get_vector
from trainer import training, train_pretrain_siamese_model, init_pretrain_siamese_model
import numpy as np
import random
import pandas as pd


# All possible aldehyde and amine pairs. Please change the list according to your dataset.
POSSIBLE_ALD = ['ald_1', 'ald_7', 'ald_8', 'ald_10', 'ald_11', 'ald_13', 'ald_20', 'ald_21', 'ald_23', 'ald_24', 'ald_25']
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

    def evaluate_batch(self, batch, top_k=3, excel_path='evaluate_results.xlsx'):
        # Evaluate a batch of candidate pairs using weighted similarity scoring,
        # then save results to Excel as a 26x20 matrix (rows: amines, columns: alds)
        self.train_model()
        self.model.eval()

        # Precompute evaluated embeddings and labels
        evaluated_vectors = [get_vector(p[0], p[1]) for p in self.chosen_points]
        evaluated_labels = self.evaluations
        eval_x2 = torch.Tensor(np.array(evaluated_vectors))
        label2 = torch.Tensor(np.array(evaluated_labels))

        results = []  # store dicts: {'amine': node, 'ald': linker, 'score': rounded_score}

        with torch.no_grad():
            for pair in batch:
                if self.evaluated(pair):
                    continue

                # Compute candidate embedding
                vec1 = torch.Tensor(get_vector(pair[0], pair[1]))
                x1 = vec1.unsqueeze(0).repeat(eval_x2.size(0), 1)

                # Model inference
                if self.pretrain:
                    out1 = self.model(x1)
                    out2 = self.model(eval_x2)
                else:
                    out1, out2 = self.model(x1, eval_x2)

                # Compute squared distances and clamp
                d2 = (out2 - out1).pow(2).sum(dim=1)
                d2 = torch.clamp(d2, max=50.0)
                weights = torch.exp(-d2)

                # Select top_k weights
                if top_k < weights.size(0):
                    top_w, top_idx = torch.topk(weights, top_k)
                    sel_labels = label2[top_idx]
                    weights = top_w
                else:
                    sel_labels = label2

                # Normalize and compute score
                numerator = (sel_labels * weights).sum()
                denominator = weights.sum() + 1e-8
                score = (numerator / denominator).item()

                # Round to integer
                rounded_score = round(score)

                results.append({
                    'amine': pair[0],
                    'ald': pair[1],
                    'pred_score': rounded_score
                })

                print(f'Pair {pair} predicted score (rounded): {rounded_score}')

        # Create DataFrame and pivot to 26x20 table
        df = pd.DataFrame(results)
        df_matrix = df.pivot(index='amine', columns='ald', values='pred_score')
        df_matrix.to_excel(excel_path, index=True)
        print(f'All results saved to {excel_path} (rows=amines, cols=alds)')

    def suggest_batch(self, batch_size=3, top_k=3):
        # suggest the next batch of pairs to evaluate based on the trained model
        self.train_model()
        self.model.eval()

        pair_scores = []
        pair_name = []

        # Precompute evaluated vectors and labels
        evaluated_vectors = [get_vector(p[0], p[1]) for p in self.chosen_points]
        evaluated_labels = self.evaluations  # list of floats or tensors

        # Stack evaluated tensors once
        eval_x2 = torch.Tensor(np.array(evaluated_vectors))
        label2 = torch.Tensor(np.array(evaluated_labels))

        with torch.no_grad():
            # Process each candidate pair
            for pair in self.all_pairs:
                if self.evaluated(pair):
                    continue

                # Get embedding for candidate
                vec1 = torch.Tensor(get_vector(pair[0], pair[1]))
                # Expand to match number of evaluated points
                x1 = vec1.unsqueeze(0).repeat(eval_x2.size(0), 1)

                # Model forward
                if self.pretrain:
                    out1 = self.model(x1)
                    out2 = self.model(eval_x2)
                else:
                    out1, out2 = self.model(x1, eval_x2)

                # Compute squared distances
                d2 = (out2 - out1).pow(2).sum(dim=1)
                # Clamp for numerical stability
                d2 = torch.clamp(d2, max=50.0)

                # Compute Gaussian-like weights
                weights = torch.exp(-d2)

                # Keep only top_k largest weights
                if top_k < weights.size(0):
                    top_weights, top_idx = torch.topk(weights, top_k)
                    selected_labels = label2[top_idx]
                    weights = top_weights
                else:
                    selected_labels = label2

                # Weighted normalization and scoring
                numerator = (selected_labels * weights).sum()
                denominator = weights.sum() + 1e-8  # avoid div zero
                score = (numerator / denominator).item()

                pair_name.append(pair)
                pair_scores.append(score)

        # Convert to numpy array for sorting
        scores_arr = np.array(pair_scores)
        # Select best batch_size pairs
        chosen_idx = np.argsort(scores_arr)[-batch_size:]
        chosen_pairs = [pair_name[i] for i in chosen_idx]
        chosen_scores = scores_arr[chosen_idx]

        return chosen_pairs, chosen_scores
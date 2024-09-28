from trainer import init_pretrain_siamese_model
import torch
from torch import nn
from data_preprocess import get_vector
import shap
import matplotlib.pyplot as plt


pretrain_model = init_pretrain_siamese_model()

ald_list1 = ['ald_1', 'ald_20', 'ald_7', 'ald_8', 'ald_13']
amine_list1 = ['amine_11', 'amine_11', 'amine_11', 'amine_12', 'amine_8']
result1 = [0.0003, 0.001, 0.005, 0.025, 0.106]

ald_list2 = ['ald_25', 'ald_13', 'ald_23', 'ald_25', 'ald_13', 'ald_13']
amine_list2 = ['amine_10', 'amine_10', 'amine_12', 'amine_12', 'amine_12', 'amine_11']
result2 = [0.057, 0.094, 0.11, 0.186, 0.21, 0.413]

ald_list3 = ['ald_25', 'ald_25', 'ald_13', 'ald_10', 'ald_25', 'ald_8', 'ald_8', 'ald_11', 'ald_21', 'ald_23', 'ald_24']
amine_list3 = ['amine_8', 'amine_17', 'amine_15', 'amine_11', 'amine_11', 'amine_17', 'amine_11', 'amine_11',
               'amine_11', 'amine_11', 'amine_11']
result3 = [0.001, 0.0017, 0.0018, 0.004, 0.005, 0.005, 0.018, 0.018, 0.02, 0.022, 0.071]

feature_names = [
    'ex_energy_ald',
    'fosc_ald',
    'sr_index_ald',
    'd_index_ald',
    'delta_sigma_ald',
    'h_index_ald',
    't_index_ald',
    'hdi_ald',
    'edi_ald',
    'vip_ald',
    'vea_ald',
    'mulliken_negativity_ald',
    'hardness_ald',
    'elec_index_ald',
    'ex_energy_amine',
    'fosc_amine',
    'sr_index_amine',
    'd_index_amine',
    'delta_sigma_amine',
    'h_index_amine',
    't_index_amine',
    'hdi_amine',
    'edi_amine',
    'vip_amine',
    'vea_amine',
    'mulliken_negativity_amine',
    'hardness_amine',
    'elec_index_amine'
]


def get_data(ald_list, amine_list):
    pairs = []
    pair_vectors = []
    for pair in zip(ald_list, amine_list):
        pairs.append(pair)
        pair_vectors.append(get_vector(*pair))
    return pairs, pair_vectors


def get_all_dataset():
    pairs_1, pair_vectors_1 = get_data(ald_list1, amine_list1)
    pairs_2, pair_vectors_2 = get_data(ald_list2, amine_list2)
    pairs_3, pair_vectors_3 = get_data(ald_list3, amine_list3)
    pair_vectors_3 = pair_vectors_1 + pair_vectors_2 + pair_vectors_3
    pairs_3 = pairs_1 + pairs_2 + pairs_3
    pair_vectors_2 = pair_vectors_1 + pair_vectors_2
    pairs_2 = pairs_1 + pairs_2
    pair_vectors_1 = torch.Tensor(pair_vectors_1)
    pair_vectors_2 = torch.Tensor(pair_vectors_2)
    pair_vectors_3 = torch.Tensor(pair_vectors_3)
    target_1 = torch.Tensor(result1)
    target_2 = torch.Tensor(result1 + result2)
    target_3 = torch.Tensor(result1 + result2 + result3)
    return pair_vectors_1, target_1, pair_vectors_2, target_2, pair_vectors_3, target_3


class ShapRegressionModel(nn.Module):

    def __init__(self, pretrain_model):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.regression_head = nn.Sequential(
            nn.Linear(10, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regression_head(self.pretrain_model(x))

    def load_pretrain_state_dict(self, name='pretrain_parameters.pth'):
        self.pretrain_model.load_state_dict(torch.load(name))


pair_vectors_1, target_1, pair_vectors_2, target_2, pair_vectors_3, target_3 = get_all_dataset()


def get_shap_result(pair_vectors, target, postfix=None):
    # Get the shap values of the pretrain model

    pretrain_model = EmbeddingNet()
    shap_model = ShapRegressionModel(pretrain_model)

    def train_shap_model(model, train_x, train_y, n_epochs=200):
        lr = 1e-3
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        model.train()
        total_loss = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            prediction = model(train_x)
            loss = criterion(prediction, train_y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            message = 'Epoch: {}/{}. Train set: Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item())
            print(message)


    train_shap_model(shap_model, pair_vectors, target)

    def predict(x):
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            return shap_model(x).numpy()

    explainer = shap.DeepExplainer(shap_model, pair_vectors.float())
    shap_values = explainer.shap_values(pair_vectors.float()).squeeze(-1)
    shap.summary_plot(shap_values, pair_vectors.numpy(), feature_names=feature_names, show=False)

    plt.savefig("shap_summary_plot{}.png".format(postfix), dpi=300, bbox_inches='tight')
    pd.DataFrame(shap_values).to_csv('shap_values{}.csv'.format(postfix))


get_shap_result(pair_vectors_1, target_1, postfix='_1')
get_shap_result(pair_vectors_2, target_2, postfix='_2')
get_shap_result(pair_vectors_3, target_3, postfix='_3')

import pandas as pd
import numpy as np


def get_normalized_descriptors():
    df = pd.read_csv('full_descriptors.csv')
    numeric_cols = df.select_dtypes(include=[np.number])
    normalized_df = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
    df[numeric_cols.columns] = normalized_df
    df['fchknam'] = df['fchknam'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    df = df.set_index('fchknam')
    return df, numeric_cols.mean(), numeric_cols.std()


def get_vector(item1, item2):
    df, _, _ = get_normalized_descriptors()
    return np.concatenate([df.loc[item1].to_numpy(), df.loc[item2].to_numpy()])


def pretraining_data_preprocess():
    df = pd.read_csv('removed_combines_hecontri.txt')
    df['ald'] = df['fname'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    df['amine'] = df['fname'].apply(lambda x: x.split('_')[4] + '_' + x.split('_')[5])
    data, _, _ = get_normalized_descriptors()
    df['ald'] = df['fname'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
    df['amine'] = df['fname'].apply(lambda x: x.split('_')[4] + '_' + x.split('_')[5])
    train_data = pd.DataFrame([])
    for i in range(len(df)):
        row = df.iloc[i]
        ald = data.loc[row['ald']]
        amine = data.loc[row['amine']]
        train_data = train_data.append(pd.concat([ald, amine]), ignore_index=True)

    return train_data, df['amine_hole_contri'].apply(lambda x: eval(x[:-1]) / 100), df['amine_elec_contri'].apply(
        lambda x: eval(x[:-1]) / 100)

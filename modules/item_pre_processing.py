import feather
import pandas
import pandas as pd
import item_model
import numpy as np


def normalize_features(df):
    for col in df.columns:
        col_max = df[col].max() + 1
        df[col] = df.apply(lambda row: (row[col] + 1) / col_max, axis=1)
    return df


def df_to_np_input(df):
    inp_brand = np.array([[float(value)] for value in list(df['brand'].values)])
    inp_feature_1 = np.array([[float(value)] for value in list(df['feature_1'].values)])
    inp_feature_2 = np.array([[float(value)] for value in list(df['feature_2'].values)])
    inp_feature_3 = np.array([[float(value)] for value in list(df['feature_3'].values)])
    inp_feature_4 = np.array([[float(value)] for value in list(df['feature_4'].values)])
    inp_feature_5 = np.array([[float(value)] for value in list(df['feature_5'].values)])
    return inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5


def df_to_np_out(df):
    out_item = df['itemID'].values
    return out_item


def mse(actual, pred):
    # actual, pred are np arrays
    return np.square(np.subtract(actual, pred)).mean()


def guess_item(df, item_features, count=5):
    # item_features: list of normalized features (0..1)
    # count: the best [count] guesses will be returned as df
    item_features = np.array(item_features)
    df['MSE'] = df.apply(lambda row: mse(np.array([row['brand'], row['feature_1'], row['feature_2'], row['feature_3'],
                                                   row['feature_4'], row['feature_5']]), item_features), axis=1)
    guesses = df.nsmallest(n=count, columns=['MSE'])

    return guesses


if __name__ == '__main__':
    # items = pd.read_feather('../resources/items_features_f.feather')
    # items = normalize_features(items)
    # items.to_feather('../resources/items_nom_features_f.feather')
    items = pd.read_feather('../resources/items_nom_features_f.feather')
    items1 = pd.read_feather('../resources/items_features_f.feather')
    df = guess_item(items, [0.56935, 0.45455, 0.25, 0.91095, 0.6, 0.35079])

    items1 = items1.sort_values(by=['itemID'])
    # 22665|861|4|0|490|2|66
    # 16557|366|6|2|497|0|13
    inp_brand, inp_feature_1, inp_feature_2, inp_feature_3, inp_feature_4, inp_feature_5 = df_to_np_input(items)
    out_item = df_to_np_out(items)

    # get item model
    ItemPredictor = item_model.ItemModel(inp_brand,
                                         inp_feature_1,
                                         inp_feature_2,
                                         inp_feature_3,
                                         inp_feature_4,
                                         inp_feature_5,
                                         out_item)
    # create model
    ItemPredictor.create_model()
    ItemPredictor.get_model_info()
    # compile
    ItemPredictor.compiling()
    # train item model
    ItemPredictor.fit(inp_brand,
                      inp_feature_1,
                      inp_feature_2,
                      inp_feature_3,
                      inp_feature_4,
                      inp_feature_5,
                      out_item)
    # test item model
    first_row = items.loc[0]
    inp_brand_t = first_row['brand']
    inp_feature_1_t = first_row['feature_1']
    inp_feature_2_t = first_row['feature_2']
    inp_feature_3_t = first_row['feature_3']
    inp_feature_4_t = first_row['feature_4']
    inp_feature_5_t = first_row['feature_5']
    out_item_t = first_row['itemID']

    res = ItemPredictor.testing(inp_brand_t, inp_feature_1_t,
                          inp_feature_2_t, inp_feature_3_t, inp_feature_4_t, inp_feature_5_t, max_value=32775 + 1)


    pass
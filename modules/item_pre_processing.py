import feather
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import item_model
import numpy as np
from graphs import GraphVisualization
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import NearestNeighbors
import time


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


def create_user_item_matrix():
    orders = pd.read_csv('../resources/orders.csv', delimiter='|')
    items = pd.read_csv('../resources/items.csv', delimiter='|')
    orders = orders[['userID', 'itemID', 'order']]
    group = orders.groupby(['userID', 'itemID'])['order'].sum()
    orders_no_duplicates = pd.DataFrame(group).reset_index()

    time_now = time.time()
    user_data = []
    item_data = []
    n_orders = []
    for i in range(orders_no_duplicates.shape[0]):  # 1071538
        user = (orders['userID'].iloc[i])
        item = (orders['itemID'].iloc[i])
        order = (orders['order'].iloc[i])
        user_data.append(user)
        item_data.append(item)
        n_orders.append(order)

    adj_matrix = csr_matrix((n_orders, (user_data, item_data)))
    csr_shape = adj_matrix.shape

    U, Sigma, VT = randomized_svd(adj_matrix, n_components=50, n_iter=5, random_state=None)
    print(f' time passed: {(time.time() - time_now) / 60} min')
    print(U.shape)
    print(Sigma.shape)
    print(VT.T.shape)

    # df = pd.DataFrame(0, index=np.arange(46138), columns=[str(i) for i in range(32776)])
    orders_group = orders.groupby(['itemID'])
    oders_group_list = [orders_group.get_group(x) for x in orders_group.groups]
    group = orders.groupby(['userID', 'itemID'])['order'].sum()
    # graphs_list = []
    for i in range(1071538):
        df[str(orders['itemID'][i])][orders['userID'][i]] += orders['order'][i]
    # df.to_csv('user_item_matrix.csv', delimiter='|')
    df.to_feather('user_item_matrix_f.feather')
    pass


def create_item_category_matrix():
    items = pd.read_csv('../resources/items.csv', delimiter='|')
    cat = pd.read_csv('../resources/category_hierarchy.csv', delimiter='|')
    G = GraphVisualization()
    iter = cat.iterrows()
    for i in iter:
        G.addEdge(i[1]['category'], i[1]['parent_category'])
    connected_nodes = {}
    for i in range(4300):
        connected_nodes.update({str(i): G.get_neighbors(i)})

    items_cat = items[['itemID', 'categories']]
    items_cat = items_cat.sort_values(['itemID'])
    items_cat = items_cat[['categories']]
    items_cat['categories'] = items_cat.apply(
        lambda row: row['categories'].strip('][').split(', ') if str(row['categories']) != 'nan' else [], axis=1)
    categories_dict = {}
    for category in range(4300):
        item_cat_list = []
        for row_categories in items_cat['categories']:
            if str(category) in row_categories:
                item_cat_list.append(1)
            else:
                item_cat_list.append(0)
        categories_dict.update({str(category): item_cat_list})

    item_idx = 0
    for row_categories in items_cat['categories']:
        parents = []
        for node in row_categories:
            parents = parents + connected_nodes[str(node)]
        for p in parents:
            if categories_dict[str(p)][item_idx] == 0:
                categories_dict[str(p)][item_idx] = 0.5
        item_idx += 1

    df = pd.DataFrame(categories_dict)
    df.to_csv('item_category_matrix.csv', sep='|')
    df.to_feather('item_category_matrix_f.feather')
    pass


def analyse_correlation_between_category_and_features():
    """checking correlation between item category and item features/brand.

    RESULTS:
    -> it seems like items in same category share about 80% of there feature values/brand.
    -> they pairwise difference in feature values/brand is only 0 to 2 features out of 6 (with brand)
    -> often one value from one feature is dominant over the other values in this feature
    (dominant means ~70% of the values in that features are the same)
    """
    # first do this analisation on one item
    items = pd.read_csv('../resources/items.csv', delimiter='|').sort_values(['itemID'])

    items['categories'] = items.apply(
        lambda row: row['categories'].strip('][').split(', ') if str(row['categories']) != 'nan' else [], axis=1)
    items = items.explode("categories")
    group = items.groupby(['categories'])
    keys = group.groups.keys()
    cat_groups = [group.get_group(cat) for cat in keys]
    # analyse results
    cat_1000 = cat_groups[2]        # select example category group -> here category 1000
    cat_1000_brand = pd.DataFrame(cat_1000.groupby(['brand'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_1 = pd.DataFrame(cat_1000.groupby(['feature_1'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_2 = pd.DataFrame(cat_1000.groupby(['feature_2'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_3 = pd.DataFrame(cat_1000.groupby(['feature_3'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_4 = pd.DataFrame(cat_1000.groupby(['feature_4'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_5 = pd.DataFrame(cat_1000.groupby(['feature_5'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    # plot feature values/brand against there occurrences
    cat_1000_brand.plot()
    cat_1000_feature_1.plot()
    cat_1000_feature_2.plot()
    cat_1000_feature_3.plot()
    cat_1000_feature_4.plot()
    cat_1000_feature_5.plot()
    plt.show()
    pass


if __name__ == '__main__':
    analyse_correlation_between_category_and_features()


    create_user_item_matrix()

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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import feather
from modules.encoder import Encoder
from modules.date_handler import DateHandler


def displaying_csv_pairplots(df):
    """plot df in a pair plot.
    """
    sns.pairplot(df, height=2, plot_kws={"s": 3})
    plt.tight_layout()
    plt.show()
    pass


def correlations(df):
    """find linear correlations between the column series of df,
    plot them as heatmap
    """
    labels = list(df.columns.values)
    cm = np.corrcoef(df.values.T)
    sns.set(font_scale=1.5)
    # plot correlations as heatmap

    heat_map1 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
                            yticklabels=labels, xticklabels=labels)
    plt.show()
    pass


def extract_dates_information():
    df2 = pd.read_csv('../resources/orders.csv', delimiter='|')
    df2['date'] = df2.apply(lambda row: pd.to_datetime(row['date']), axis=1)
    D = DateHandler()
    df2 = D.date_to_day(df2, col_name='date')
    df2 = D.date_to_month(df2, col_name='date')
    df2 = D.date_to_day_of_month(df2, col_name='date')

    df2.to_csv('../resources/one_hot_dates' + '.csv', sep='|', index=False)
    df2.to_feather('../resources/one_hot_dates_f' + '.feather')


def filter_trend_and_seasonality(df):
    from statsmodels.tsa.seasonal import seasonal_decompose

    df = df[['date', 'order']]
    df['date'] = df.apply(lambda row: pd.to_datetime(row['date']), axis=1)
    df = df.set_index('date')
    # Multiplicative Decomposition
    # result_mul = seasonal_decompose(df, model='multiplicative')

    # Additive Decomposition
    result_add = seasonal_decompose(df, model='additive', extrapolate_trend='freq')

    # Plot
    plt.rcParams.update({'figure.figsize': (10, 10)})
    # result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
    result_add.plot().suptitle('Additive Decompose', fontsize=22)

    plt.show()
    return result_add.resid, result_add.trend, result_add.seasonal

# First networkx library is imported
# along with matplotlib
import networkx as nx


# Defining a Class
class GraphVisualization:

    def __init__(self):

        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # def get_nodes(self):



    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self, labels):
        G_ = nx.Graph()
        G_.add_edges_from(self.visual)
        return list(self.G.nodes)
        # nx.draw_networkx(G_, with_labels=True, node_size=10, labels=labels, font_size=15, alpha=0.5,
        #                  horizontalalignment='right',
        #                  verticalalignment='bottom')
        # plt.show()


if __name__ == '__main__':
    item_cat = pd.read_feather('../modules/item_category_matrix_f.feather')

    sns.heatmap(item_cat)
    plt.show()

    ord = pd.read_csv('../resources/orders.csv', delimiter='|')
    # cat.plot(x='category', y='parent_category', kind='scatter', s=3)
    # Driver code
    cat = None

    cat_group = cat.groupby(by='parent_category')
    graphs_list = [cat_group.get_group(x) for x in cat_group.groups]
    G = []
    labels = []
    idx = 0
    g_i = 0
    labels.append({})
    G.append(GraphVisualization())
    name = 'category_graph'
    for g in graphs_list:
        iter = g.iterrows()
        for i in iter:
            labels[g_i].update({i[1]['parent_category']: i[1]['parent_category']})
            G[g_i].addEdge(i[1]['category'], i[1]['parent_category'])
        if idx % 15 == 0 and idx != 0:
            G[g_i].visualize(labels=labels[g_i])
            G.append(GraphVisualization())
            labels.append({})
            g_i += 1
            plt.savefig('../figures/' + name + str(g_i) + '.png')
            plt.clf()
        idx += 1
    if (idx - 1) % 15 != 0:
        G[g_i].visualize(labels=labels[g_i])
        G.append(GraphVisualization())
        plt.savefig('../figures/' + name + str(g_i) + '.png')
        plt.clf()

    df = pd.read_feather('../resources/orders_date_f.feather')[['date', 'day_of_month', 'month', 'n_day_of_week', 'userID', 'itemID', 'order']]
    # cols = ['itemID', 'userID', 'n_day_of_week', 'day_of_month']
    # cols = ['itemID', 'date']
    # correlations(df[cols])
    # displaying_csv_pairplots(df[cols]) # 395
    # df[cols].plot(x='date', y='itemID', kind='scatter', s=3)
    # plt.show()
    # OneHot = Encoder(code_type='one_hot')
    # DateEncoder = OneHot.date_encoder(df2)


    # df = pd.read_csv('../resources/items.csv', delimiter='|')
    # df = df.sort_values(by=['date', 'userID', 'itemID'])
    df = df[['date', 'order', 'userID']]
    # df = df.groupby(df['userID']).sum()
    # df = df.reset_index()
    # df = df.reset_index()
    # fig = df.plot(x='index', y='order', kind='line')
    # plt.show()
    df = df[df['userID'].isin([38604])]
    df = df.drop('userID', axis=1)
    df = df.groupby(['date']).sum()
    df = df.reset_index()
    r = pd.date_range(start=df.date.min(), end=df.date.max())
    df = df.set_index('date').reindex(r).fillna(0.0).rename_axis('date').reset_index()

    resid, trend, seasonality = filter_trend_and_seasonality(df)
    df = df.reset_index()
    fig = df.plot(x='index', y='order', kind='line')
    plt.show()

    # df = df.reset_index()
    # df = df[['n_day_of_week', 'itemID']]
    # df = pd.concat([df] * 4, ignore_index=True)
    # df = df.reset_index()
    # # correlations(df)
    # fig = df.plot(x='index', y='itemID', kind='line')
    # plt.show()
    pass



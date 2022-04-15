import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import feather
from modules.encoder import Encoder
from modules.date_handler import DateHandler


def displaying_csv_pairplots():
    """load dataset, define plotted columns and plot them in a pair plot.
    First plot the items.csv
    Second plot the orders.csv
    """
    df = pd.read_csv('../resources/items.csv', delimiter='|')
    cols = ['itemID', 'brand', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    sns.pairplot(df[cols], height=2, plot_kws={"s": 3})
    plt.tight_layout()
    plt.show()

    df2 = pd.read_csv('../resources/orders.csv', delimiter='|', parse_dates=True)
    first_date = pd.to_datetime(df2['date'][0]).value
    df2['date'] = df2.apply(lambda row: pd.to_datetime(row['date']).value - first_date, axis=1)
    cols2 = ['date', 'userID', 'itemID', 'order']
    sns.pairplot(df2[cols2], height=2.5, plot_kws={"s": 3})
    plt.tight_layout()
    plt.show()


def correlations(df):
    """find linear correlations
    plot correlations of items.csv and orders.csv
    """
    # plot correlations between the features of items.csv
    df = pd.read_csv('../resources/items.csv', delimiter='|')
    cols = ['itemID', 'brand', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    heat_map1 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
                            yticklabels=cols, xticklabels=cols)
    plt.show()

    # plot correlations between features of orders.csv
    df2 = pd.read_csv('../resources/orders.csv', delimiter='|', parse_dates=True)
    first_date = pd.to_datetime(df2['date'][0]).value
    df2['date'] = df2.apply(lambda row: pd.to_datetime(row['date']).value - first_date, axis=1)
    cols2 = ['date', 'userID', 'itemID', 'order']
    cm = np.corrcoef(df2[cols2].values.T)
    sns.set(font_scale=1.5)
    heat_map2 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
                            yticklabels=cols2, xticklabels=cols2)
    plt.show()
    pass


if __name__ == '__main__':
    correlations(df)
    # df2 = pd.read_csv('../resources/orders.csv', delimiter='|')
    # # OneHot = Encoder(code_type='one_hot')
    # # DateEncoder = OneHot.date_encoder(df2)
    #
    # df2['date'] = df2.apply(lambda row: pd.to_datetime(row['date']), axis=1)
    # D = DateHandler()
    # df2 = D.date_to_day(df2, col_name='date')
    # df2 = D.date_to_month(df2, col_name='date')
    # df2 = D.date_to_day_of_month(df2, col_name='date')
    #
    # df2.to_csv('../resources/one_hot_dates' + '.csv', sep='|', index=False)
    # df2.to_feather('../resources/one_hot_dates_f' + '.feather')

    pass



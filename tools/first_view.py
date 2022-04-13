import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import feather
from modules.encoder import Encoder


def displaying_csv_pairplots():
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


def correlations():
    """find linear correlations"""

    df = pd.read_csv('../resources/items.csv', delimiter='|')
    cols = ['itemID', 'brand', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    heat_map1 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},
                            yticklabels=cols, xticklabels=cols)
    plt.show()

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
    # correlations()
    df2 = pd.read_csv('../resources/orders.csv', delimiter='|', parse_dates=True)
    OneHot = Encoder(code_type='one_hot')
    DateEncoder = OneHot.date_encoder(df2)

    pass



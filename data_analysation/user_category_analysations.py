import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def correlations_between_categories_a_user_order(mapped_order_item_feature_category, items):
    # extract categories from list
    mapped_order_item_feature_category['categories'] = mapped_order_item_feature_category['categories'].apply(
        lambda cat: cat.strip('][').split(', ') if str(cat) != 'nan' else [])
    mapped_order_item_feature_category_all_cat = mapped_order_item_feature_category.explode("categories")
    # sort by the top most liked categories for every user
    categories_user_group = mapped_order_item_feature_category_all_cat.groupby(['userID', 'categories']).count().reset_index().sort_values(['userID', 'order'], ascending=False).reset_index()[['userID', 'categories', 'order']]

    # calculate: "chance that a user order another item from the same category,
    # if the user already ordered an item from this category"
    # Todo: for now this metric will take to long
    # categories_user_group['category_chance'] = categories_user_group.apply(lambda row: row['order'] / mapped_order_item_feature_category[mapped_order_item_feature_category['userID'] == row['userID']].shape[0], axis=1)

    # extract categories from list
    items['categories'] = items['categories'].apply(
        lambda cat: cat.strip('][').split(', ') if str(cat) != 'nan' else [])
    items = items.explode("categories")

    # get itemsIDs for each category
    grp = items.groupby(['categories'], as_index=False).aggregate({
        'itemID': [list],
    })
    grp.columns = ['category', 'itemIDs']
    grp = grp.explode('itemIDs').reset_index(drop=True)

    # get the highest ranked category for each user
    # list userIDs
    user_grps = {'userID': [], 'highest_categories': []}
    for user_id in range(46138):
        user_id_df = categories_user_group[categories_user_group['userID'] == user_id]
        if user_id_df.shape[0] > 0:
            user_grps['userID'].append(user_id_df['userID'].max())
            user_grps['highest_categories'].append(user_id_df['categories'].tolist()[0])
    user_grps_df = pd.DataFrame(user_grps)

    highest_cats_count = user_grps_df.groupby(['highest_categories']).count().reset_index().sort_values(['userID'], ascending=False).reset_index(drop=True)
    # get top items from these categories to later recommend these to the specific users

    pass


if __name__ == '__main__':
    mapped_order_item_feature_category = pd.read_csv('../resources/mapped_order_item_features_category.csv', delimiter='|')
    items = pd.read_csv('../resources/items.csv', delimiter='|')

    correlations_between_categories_a_user_order(mapped_order_item_feature_category, items)
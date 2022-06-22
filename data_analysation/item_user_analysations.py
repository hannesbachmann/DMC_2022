import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def total_reorders_count_for_every_item(orders):
    """total number of reorders for every item
    - mean, total
    """
    # how often every item gets ordered from the same user - every item order count as 1
    reorder_per_user = orders.groupby(['userID', 'itemID']).count().reset_index()[['userID', 'itemID', 'order']].rename(
        columns={'order': 'reorder'})
    # -1 to get the correct reorder count
    reorder_per_user['reorder'] = reorder_per_user['reorder'] - 1
    # filter all reordered items
    reorder_per_user = reorder_per_user[reorder_per_user['reorder'] > 0]
    # count total numbers of reorders for every item
    reorders = reorder_per_user.groupby(['itemID']).sum().reset_index()[['itemID', 'reorder']].rename(
        columns={'reorder': 'reorder_total'})
    reorders['reorder_mean'] = reorders.apply(lambda row: row['reorder_total'] / 46138, axis=1)

    return reorders.sort_values(['itemID']).sort_values(['reorder_total'], ascending=False).reset_index(drop=True)


def reorder_count_same_user(orders):
    """how often a user reorder an item"""
    orders['tmp_count'] = 1
    # how often every item gets ordered from the same user - every item order count as 1
    reorder_per_user = orders.groupby(['userID', 'itemID']).sum().reset_index()[['userID', 'itemID', 'order']].rename(
        columns={'order': 'reorder'})
    # -1 to get the correct reorder count
    reorder_per_user['reorder'] = reorder_per_user['reorder'] - 1

    return reorder_per_user


def number_of_orders_same_user(orders):
    """how often an item gets ordered from the same user
    """
    # how often every item gets ordered from the same user - every item order count as 1
    reorder_per_user = orders.groupby(['userID', 'itemID']).sum().reset_index()[['userID', 'itemID', 'order']]

    return reorder_per_user.sort_values(['userID', 'order']).reset_index(drop=True)


def item_order_count(orders):
    """how often every item gets ordered in general (total order count for every item)
    """
    # count all orders for every item
    total_orders_per_item = orders.groupby(['itemID']).sum().reset_index()[['itemID', 'order']].rename(
        columns={'order': 'total_order_count'}).sort_values(['total_order_count'], ascending=False).reset_index(
        drop=True)

    return total_orders_per_item


def count_users_that_order_item(orders):
    """how many users order an item
    """
    orders = orders[['userID', 'itemID', 'order']].sort_values(['userID', 'itemID'])
    # combine all duplications of item-user pairs - how often a user order an item
    orders_group = orders.groupby(['userID', 'itemID']).sum().reset_index()

    # how many users order an item - only one order per user counts
    items_group = orders_group.groupby(['itemID']).count().sort_values(['order'], ascending=False).reset_index()

    return items_group.rename(columns={'order': 'order_count'})[['itemID', 'order_count']]


def count_users_that_reorder_item(orders):
    """how many users reorder a specific item
    count users that reorder item
    """
    # add counter that counts the number of reorders per item for every user
    orders['tmp_count'] = 1
    # combine all duplications of item-user pairs - how often a user order an item
    orders_group = orders.groupby(['userID', 'itemID']).sum().reset_index()
    # filter items that gets reordered
    reordered_items = orders_group[orders_group['tmp_count'] > 1]

    # how many users reorder an item
    reordered_items = reordered_items.groupby(['itemID']).count().sort_values(['order'], ascending=False).reset_index()

    return reordered_items.rename(columns={'tmp_count': 'reorder_count'})[['itemID', 'reorder_count']]


def total_count_of_users_that_reorder_items(orders):
    """how many users reorder items - total number
    """
    # add counter that counts the number of reorders per item for every user
    orders['tmp_count'] = 1
    # combine all duplications of item-user pairs - how often a user order an item
    orders_group = orders.groupby(['userID', 'itemID']).sum().reset_index()
    # filter items that gets reordered
    reordered_items = orders_group[orders_group['tmp_count'] > 1]

    reorder_count_group = reordered_items.groupby(['userID']).count().reset_index()
    # filter users that reorder items
    users_reorder_items = reorder_count_group[reorder_count_group['tmp_count'] > 0]

    return reorder_count_group.shape[0]


def correlations_between_items_a_user_order(orders, items):
    """find correlations between feature-/brand-values from all items one user buy

    ° analysing how often the same brand occur for one user in different items -> is the user *brand affine* ?
    """
    # combine all duplications of item-user pairs - how often a user order an item
    orders_group = orders.groupby(['userID', 'itemID']).sum().reset_index()

    # map features/brand to items in orders
    items = items.set_index(['itemID'])
    features_brand_category = {'brand': [],
                               'feature_1': [],
                               'feature_2': [],
                               'feature_3': [],
                               'feature_4': [],
                               'feature_5': [],
                               'categories': []}
    for item_id in orders_group['itemID']:
        features_brand_category['brand'].append(items['brand'][item_id])
        features_brand_category['feature_1'].append(items['feature_1'][item_id])
        features_brand_category['feature_2'].append(items['feature_2'][item_id])
        features_brand_category['feature_3'].append(items['feature_3'][item_id])
        features_brand_category['feature_4'].append(items['feature_4'][item_id])
        features_brand_category['feature_5'].append(items['feature_5'][item_id])
        features_brand_category['categories'].append(items['categories'][item_id])
    features_brand_df = pd.DataFrame(features_brand_category)
    mapped_order_item_features_category = orders_group
    mapped_order_item_features_category['brand'] = features_brand_df['brand']
    mapped_order_item_features_category['feature_1'] = features_brand_df['feature_1']
    mapped_order_item_features_category['feature_2'] = features_brand_df['feature_2']
    mapped_order_item_features_category['feature_3'] = features_brand_df['feature_3']
    mapped_order_item_features_category['feature_4'] = features_brand_df['feature_4']
    mapped_order_item_features_category['feature_5'] = features_brand_df['feature_5']
    mapped_order_item_features_category['categories'] = features_brand_df['categories']

    mapped_order_item_features_category.to_csv('mapped_order_item_features_category_train_all_except_one.csv', sep='|')

    return mapped_order_item_features_category


def shared_features_per_user(orders_items_features):
    """find shared features-/brand-values a user order
    ° share_brand = (number_of_all_order_with_same_brand_per_user / number_of_orders_per_user)
    """
    # how often a feature-/brand-value gets ordered from the same user
    brand_count_per_user = \
        orders_items_features.groupby(['userID', 'brand']).sum().reset_index().sort_values(['userID', 'order'])[
            ['userID', 'brand', 'order']].reset_index(drop=True)
    feature_1_count_per_user = \
        orders_items_features.groupby(['userID', 'feature_1']).sum().reset_index().sort_values(['userID', 'order'])[
            ['userID', 'feature_1', 'order']].reset_index(drop=True)
    feature_2_count_per_user = \
        orders_items_features.groupby(['userID', 'feature_2']).sum().reset_index().sort_values(['userID', 'order'])[
            ['userID', 'feature_2', 'order']].reset_index(drop=True)
    feature_3_count_per_user = \
        orders_items_features.groupby(['userID', 'feature_3']).sum().reset_index().sort_values(['userID', 'order'])[
            ['userID', 'feature_3', 'order']].reset_index(drop=True)
    feature_4_count_per_user = \
        orders_items_features.groupby(['userID', 'feature_4']).sum().reset_index().sort_values(['userID', 'order'])[
            ['userID', 'feature_4', 'order']].reset_index(drop=True)
    feature_5_count_per_user = \
        orders_items_features.groupby(['userID', 'feature_5']).sum().reset_index().sort_values(['userID', 'order'])[
            ['userID', 'feature_5', 'order']].reset_index(drop=True)

    # how often a user orders items
    items_ordered_per_user = orders_items_features.groupby(['userID']).sum().reset_index()[['userID', 'order']]

    # calculate share_brand/share_feature with above metric
    brand_count_per_user['share_brand'] = brand_count_per_user.apply(
        lambda row: row['order'] / items_ordered_per_user['order'][row['userID']], axis=1)
    feature_1_count_per_user['share_feature_1'] = feature_1_count_per_user.apply(
        lambda row: row['order'] / items_ordered_per_user['order'][row['userID']], axis=1)
    feature_2_count_per_user['share_feature_2'] = feature_2_count_per_user.apply(
        lambda row: row['order'] / items_ordered_per_user['order'][row['userID']], axis=1)
    feature_3_count_per_user['share_feature_3'] = feature_3_count_per_user.apply(
        lambda row: row['order'] / items_ordered_per_user['order'][row['userID']], axis=1)
    feature_4_count_per_user['share_feature_4'] = feature_4_count_per_user.apply(
        lambda row: row['order'] / items_ordered_per_user['order'][row['userID']], axis=1)
    feature_5_count_per_user['share_feature_5'] = feature_5_count_per_user.apply(
        lambda row: row['order'] / items_ordered_per_user['order'][row['userID']], axis=1)
    pass


def distribution_of_dominant_features_per_user(orders_items_features):
    """distribution on every brand-/feature-value per user
    ° distribution_brand = (number_of_different_items_with_same_brand_a_user_order / number_of_different_items_a_user_order)
    ° high distribution_brand value means a user order lots of different items with the same brand
    ° e.g. find out if the user order many items from the same brand in different orders

    ° it seems like:
    -> the average user dont has a favorite brand, but some brands are rare
    -> 4, 6, 10 are dominant feature_1 values for the most users, the rest is rare (e.g. 0, 2, -1)
    -> 0 is dominat feature_2 value for the most users
    -> the average user dont has a favorite feature_3 value, but some feature_3 values are in general dominant e.g. 503
    -> 0 and 3 are dominant feature_4 values for the most users (1, 2 and 4 are rare)
    ° rare brand-/feature-values can be put on a blacklist
    """
    # reset order values to 1
    orders_items_features['order'] = 1

    # number_of_different_items_with_same_brand_a_user_order
    same_brand_different_items = orders_items_features.groupby(['userID', 'brand']).sum().reset_index()[
        ['userID', 'brand', 'order']]
    same_feature_1_different_items = orders_items_features.groupby(['userID', 'feature_1']).sum().reset_index()[
        ['userID', 'feature_1', 'order']]
    same_feature_2_different_items = orders_items_features.groupby(['userID', 'feature_2']).sum().reset_index()[
        ['userID', 'feature_2', 'order']]
    same_feature_3_different_items = orders_items_features.groupby(['userID', 'feature_3']).sum().reset_index()[
        ['userID', 'feature_3', 'order']]
    same_feature_4_different_items = orders_items_features.groupby(['userID', 'feature_4']).sum().reset_index()[
        ['userID', 'feature_4', 'order']]
    same_feature_5_different_items = orders_items_features.groupby(['userID', 'feature_5']).sum().reset_index()[
        ['userID', 'feature_5', 'order']]

    # number_of_different_items_a_user_order
    different_items_per_user = orders_items_features.groupby(['userID']).sum().reset_index()[
        ['userID', 'order']].set_index(['userID'])

    # calculate distribution of brand-/feature-values of the user
    same_brand_different_items['distribution_brand'] = same_brand_different_items.apply(
        lambda row: row['order'] / different_items_per_user['order'][row['userID']], axis=1)
    same_feature_1_different_items['distribution_feature_1'] = same_feature_1_different_items.apply(
        lambda row: row['order'] / different_items_per_user['order'][row['userID']], axis=1)
    same_feature_2_different_items['distribution_feature_2'] = same_feature_2_different_items.apply(
        lambda row: row['order'] / different_items_per_user['order'][row['userID']], axis=1)
    same_feature_3_different_items['distribution_feature_3'] = same_feature_3_different_items.apply(
        lambda row: row['order'] / different_items_per_user['order'][row['userID']], axis=1)
    same_feature_4_different_items['distribution_feature_4'] = same_feature_4_different_items.apply(
        lambda row: row['order'] / different_items_per_user['order'][row['userID']], axis=1)
    same_feature_5_different_items['distribution_feature_5'] = same_feature_5_different_items.apply(
        lambda row: row['order'] / different_items_per_user['order'][row['userID']], axis=1)
    return same_brand_different_items, \
           same_feature_1_different_items, \
           same_feature_2_different_items, \
           same_feature_3_different_items, \
           same_feature_4_different_items, \
           same_feature_5_different_items


def number_of_items_at_once(orders):
    """how many (same) items gets ordered at one time from same user, mean
    ° find out if some items get only/mostly ordered in large amounts at a time

    ° min <= mean <= max
    ° it seems like the maximum number of items a user can order at once is 100
    ° mean near min: more low order counts      | mean >= (min + max) / 2   (1)
    ° mean near max: more high order counts     | mean < (min + max) / 2    (0)
    """
    items_at_once = orders.groupby(['itemID']).mean().reset_index().rename(columns={'order': 'mean'})
    items_at_once['max'] = orders.groupby(['itemID']).max().reset_index()['order']
    items_at_once['min'] = orders.groupby(['itemID']).min().reset_index()['order']
    items_at_once['sum'] = orders.groupby(['itemID']).sum().reset_index()['order']
    items_at_once['k'] = items_at_once.apply(lambda row: 1 if (row['min'] + row['max']) / 2 >= row['mean'] else 0,
                                             axis=1)

    items_at_once = items_at_once.sort_values(['sum'], ascending=False).reset_index()[
        ['itemID', 'mean', 'max', 'min', 'sum', 'k']]
    items_at_once[['max']].plot()
    items_at_once[['mean']].plot()
    items_at_once[['min']].plot()
    items_at_once[['k']].reset_index().plot.scatter(x='index', y='k')
    plt.show()

    pass


def reorder_delay_same_user(orders):
    """time that has passed between tho orders from the same user of the same item (in days)

    ° useful by predicting the min time after which we want to recommend the same item to a user again
    """
    # convert datestr to pd timestamp
    # orders['date'] = orders.apply(lambda row: pd.Timestamp(row['date']), axis=1)
    # # encode dates as sequence of orders
    # orders = orders.sort_values(['date']).reset_index(drop=True)
    # # group dates by days to create date day mapper (first day is 0)
    # day_orders = orders.groupby(['date']).count().reset_index()[['date']].reset_index().rename(columns={'index': 'day'}).set_index(['date'])
    # day_orders['day'] = day_orders['day'] + 1
    # # map day on date
    # orders['day'] = orders.apply(lambda row: day_orders['day'][row['date']], axis=1)
    # orders.to_feather('orders_mapped_days.feather')
    #
    # # filter userID, itemID -> get order days for all userID-itemID pairs
    # grp = orders.groupby(['userID', 'itemID'], as_index=False).aggregate({
    #     'date': [pd.Series.min],
    #     'day': [list, pd.Series.count],
    #     'order': [list],
    # })
    # grp.columns = ['userID', 'itemID', 'first_order_date', 'order_days', 'order_counts', 'orders']

    # load prepared df from lines above to speed up developing
    orders_delay = pd.read_feather('orders_day_delay_same_user_same_item.feather')

    # calculate min, mean reorder delay for each item a user buy
    orders_delay['order_days_delay'] = orders_delay['order_days'].apply(
        lambda days: [days[i] - days[i - 1] for i in range(1, len(days))])
    orders_delay['order_days_delay_mean'] = orders_delay['order_days_delay'].apply(
        lambda delays: sum(delays) / len(delays) if len(delays) > 0 else None)
    orders_delay['order_days_delay_min'] = orders_delay['order_days_delay'].apply(
        lambda delays: min(delays) if delays else None)
    pass


def mean_same_user_order_count(orders):
    smaller_df = orders[['itemID', 'userID', 'order']]
    ordered_items_list = orders.drop_duplicates(subset=['itemID'])['itemID'].tolist()
    order_mean = lambda i_id: orders[orders['itemID'] == i_id].groupby(['userID']).count().reset_index()[
        'order'].mean() if i_id in ordered_items_list else 0
    mean_ordered_items = [(item_id, order_mean(item_id)) for item_id in range(32776)]
    pass


if __name__ == '__main__':
    items = pd.read_csv('../resources/items.csv', delimiter='|').sort_values(['itemID'])
    orders = pd.read_csv('../resources/orders.csv', delimiter='|').sort_values(['userID'])
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    # mapped items on orders from grouped orders dataset
    # order_brand_features = pd.read_csv('../resources/mapped_order_item_features_category.csv', delimiter='|')

    # reorder_delay_same_user(orders)
    # number_of_items_at_once(orders)
    # shared_features_per_user(order_brand_features)
    order_brand_features = correlations_between_items_a_user_order(train_df, items)
    print('done')
    # brand, f_1, f_2, f_3, f_4, f_5 = distribution_of_dominant_features_per_user(order_brand_features)
    # users_that_reorder_items = total_count_of_users_that_reorder_items(orders)
    # reorder_per_user = reorder_count_same_user(orders)
    # total_reordered = total_reorders_count_for_every_item(orders)
    # total_orders_per_user = number_of_orders_same_user(orders)
    # total_order_count = item_order_count(orders)
    # users_order_count = count_users_that_order_item(orders)
    # users_reorder_count = count_users_that_reorder_item(orders)

    pass

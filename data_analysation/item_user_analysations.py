import pandas as pd
import numpy as np


def total_reorders_count_for_every_item(orders):
    """total number of reorders for every item
    - mean, total
    """
    # how often every item gets ordered from the same user - every item order count as 1
    reorder_per_user = orders.groupby(['userID', 'itemID']).count().reset_index()[['userID', 'itemID', 'order']].rename(columns={'order': 'reorder'})
    # -1 to get the correct reorder count
    reorder_per_user['reorder'] = reorder_per_user['reorder'] - 1
    # filter all reordered items
    reorder_per_user = reorder_per_user[reorder_per_user['reorder'] > 0]
    # count total numbers of reorders for every item
    reorders = reorder_per_user.groupby(['itemID']).sum().reset_index()[['itemID', 'reorder']].rename(columns={'reorder': 'reorder_total'})
    reorders['reorder_mean'] = reorders.apply(lambda row: row['reorder_total'] / 46138, axis=1)

    return reorders.sort_values(['itemID']).sort_values(['reorder_total'], ascending=False).reset_index(drop=True)


def reorder_count_same_user(orders):
    """how often a user reorder an item"""
    ones = [1 for i in range(orders.shape[0])]
    orders['tmp_count'] = pd.Series(ones)
    # how often every item gets ordered from the same user - every item order count as 1
    reorder_per_user = orders.groupby(['userID', 'itemID']).sum().reset_index()[['userID', 'itemID', 'order']].rename(columns={'order': 'reorder'})
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
    total_orders_per_item = orders.groupby(['itemID']).sum().reset_index()[['itemID', 'order']].rename(columns={'order': 'total_order_count'}).sort_values(['total_order_count'], ascending=False).reset_index(drop=True)

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
    """how many users reorder an item
    count users that reorder item
    """
    # add counter that counts the number of reorders per item for every user
    orders['tmp_count'] = orders.apply(lambda row: int(1), axis=1)
    # combine all duplications of item-user pairs - how often a user order an item
    orders_group = orders.groupby(['userID', 'itemID']).sum().reset_index()
    # filter items that gets reordered
    reordered_items = orders_group[orders_group['tmp_count'] > 1]

    # how many users reorder an item
    reordered_items = reordered_items.groupby(['itemID']).count().sort_values(['order'], ascending=False).reset_index()

    return reordered_items.rename(columns={'tmp_count': 'reorder_count'})[['itemID', 'reorder_count']]


if __name__ == '__main__':
    # orders = pd.read_csv('../resources/orders.csv', delimiter='|').sort_values(['userID'])
    #
    # reorder_per_user = reorder_count_same_user(orders)
    # total_reordered = total_reorders_count_for_every_item(orders)
    # total_orders_per_user = number_of_orders_same_user(orders)
    # total_order_count = item_order_count(orders)
    # users_order_count = count_users_that_order_item(orders)
    # users_reorder_count = count_users_that_reorder_item(orders)

    pass

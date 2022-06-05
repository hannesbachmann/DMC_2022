import pandas as pd


def reorder_from_same_user(orders):
    """how often every item gets ordered again from the same user (in different orders, at different times)
    - mean, total
    """

    reorder_per_user = orders.groupby(['userID', 'itemID']).count().reset_index()[['userID', 'itemID', 'order']].rename(columns={'order': 'reorder'})
    reorders = reorder_per_user.groupby(['itemID']).sum().reset_index()[['itemID', 'reorder']].rename(columns={'reorder': 'reorder_total'})
    reorders['reorder_mean'] = reorders.apply(lambda row: row['reorder_total'] / 46138, axis=1)

    return reorders.sort_values(['itemID'])


def total_order_count(orders):
    """how often every item gets ordered in general (total order count for every item)
    """
    total_orders_per_item = orders.groupby(['itemID']).sum().reset_index()[['itemID', 'order']].rename(columns={'order': 'total_order_count'}).sort_values(['total_order_count']).reset_index(drop=True)

    return total_orders_per_item


if __name__ == '__main__':
    orders = pd.read_csv('../resources/orders.csv', delimiter='|').sort_values(['userID'])

    total_count_df = total_order_count(orders)
    reorders_df = reorder_from_same_user(orders)

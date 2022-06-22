import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter
import time


def chance_of_item_to_get_ordered_after_another_item_gets_ordered(orders):
    """NOT TIME AWARE
    - with reorders/without reorders
    """
    # # get users who order the item
    # user_for_item_orders = orders.groupby(['itemID']).aggregate({'userID': [list], 'order': [list]}).reset_index()
    # user_for_item_orders.columns = ['itemID', 'userIDs', 'order_counts']
    # # get items a user also order for every user who orders the item
    # also_ordered_items = lambda user_id: orders[orders['userID'] == user_id]['itemID'].tolist()
    # t = time.time()
    # user_for_item_orders['connected_to'] = user_for_item_orders.apply(
    #     lambda row: list(chain.from_iterable([also_ordered_items(u_id) for u_id in row['userIDs']])),
    #     axis=1)
    # print((time.time() - t) / 60)
    # user_for_item_orders.to_csv('user_user_item_orders_c.csv', sep='|')
    # user_for_item_orders.to_feather('user_user_item_orders_f.feather')
    user_for_item_order = pd.read_feather('user_user_item_orders_f.feather')
    user_item_distribution = user_for_item_order[['itemID', 'connected_to']]
    user_item_distribution['item_distributions'] = user_for_item_order['connected_to'].apply(
        lambda item_ids: dict(Counter(item_ids)))
    user_item_distribution['max_distribution'] = user_item_distribution['item_distributions'].apply(
        lambda item_ids: max(item_ids, key=item_ids.get))
    pass


if __name__ == '__main__':
    orders = pd.read_csv('../resources/orders.csv', delimiter='|').sort_values(['userID']).reset_index(drop=True)

    chance_of_item_to_get_ordered_after_another_item_gets_ordered(orders)

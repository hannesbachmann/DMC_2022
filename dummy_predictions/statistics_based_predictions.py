from data_analysation.item_user_analysations import reorder_count_same_user, \
    total_reorders_count_for_every_item, count_users_that_order_item, count_users_that_reorder_item
import pandas as pd
import matplotlib.pyplot as plt


def reorder_based_prediction_general_single_rec():
    """
    ° recommend the mostly reordered item to every user

    -> one vs all: ~ 0.98% correct recommendations
    """
    # orders = pd.read_csv('../resources/orders.csv', delimiter='|').sort_values(['userID'])

    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder = total_reorders_count_for_every_item(train_df)
    reorders_train_head = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True).head()

    # simply recommend the item with the most reorders to every user
    test_df['recommendation'] = test_df.apply(lambda row: reorders_train_head['itemID'][0], axis=1)
    # evaluate performance of the prediction
    test_df['validation'] = test_df.apply(lambda row: True if row['recommendation'] == row['itemID'] else False, axis=1)
    valid_groups = test_df.groupby(['validation']).count().reset_index()[['validation', 'recommendation']]
    error = (valid_groups['recommendation'][0] / valid_groups['recommendation'][1]) - 1  # min 0, lower is better
    print(f'correct recommendations: {valid_groups["recommendation"][1] / valid_groups["recommendation"][0] * 100}%')
    # -> about 0.977% correct recommendations


def reorder_based_prediction_user_specific_single_rec():
    """
    ° recommend the mostly reordered specific for every user
    -> one vs all: ~ 4.23% correct recommendations
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder_per_user = reorder_count_same_user(train_df)

    group = reorder_per_user.groupby(['userID'])
    keys = group.groups.keys()
    reorder_groups = [group.get_group(reorder_count) for reorder_count in keys]

    user_spec_reorder = {'userID': [], 'itemID': [], 'max_reorder_count': []}
    for user_id_df in reorder_groups:
        user_spec_reorder['userID'].append(user_id_df['userID'].values.max())
        max_value = int(user_id_df['reorder'].values.max())
        user_spec_reorder['max_reorder_count'].append(max_value)
        items_with_max_values = user_id_df[user_id_df['reorder'] == max_value]
        user_spec_reorder['itemID'].append(items_with_max_values['itemID'].values.max())
    rec_df = pd.DataFrame(user_spec_reorder)
    rec_df['old_itemID'] = test_df['itemID']
    # evaluate performance of the prediction
    rec_df['validation'] = rec_df.apply(lambda row: True if row['old_itemID'] == row['itemID'] else False, axis=1)
    valid_groups = rec_df.groupby(['validation']).count().reset_index()[['validation', 'itemID']]
    print(f'correct recommendations: {valid_groups["itemID"][1] / valid_groups["itemID"][0] * 100}%')
    pass


def reorder_based_prediction_user_specific_multiple_rec():
    """
    ° recommend the mostly reordered items specific for every user

    -> this goes up to ~26% for 30 items and ~28% for 100 items
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder_per_user = reorder_count_same_user(train_df)

    group = reorder_per_user.groupby(['userID'])
    keys = group.groups.keys()
    reorder_groups = [group.get_group(reorder_count).sort_values(['reorder'], ascending=False).reset_index(drop=True)
                      for reorder_count in keys]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        user_spec_reorder = {'userID': [], 'validation': []}
        for user_id_df in reorder_groups:
            user_id_item_list = user_id_df['itemID'].tolist()
            if num_recommendations > len(user_id_item_list):
                user_id_rec_list = user_id_item_list
            else:
                user_id_rec_list = user_id_item_list[:num_recommendations]
            test_item = test_df[test_df['userID'] == int(user_id_df['userID'][0])]['itemID'].tolist()[0]
            if test_item in user_id_rec_list:
                user_spec_reorder['validation'].append(True)
            else:
                user_spec_reorder['validation'].append(False)
            user_spec_reorder['userID'].append(user_id_df['userID'][0])
        new_test_df = pd.DataFrame(user_spec_reorder)

        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'userID']]
        error = (valid_groups['userID'][0] / valid_groups['userID'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["userID"][1] / valid_groups["userID"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    rec_df = pd.DataFrame(evaluation_result)
    rec_df.to_csv('reorder_based_prediction_user_specific_multiple_rec.csv', sep='|')
    pass


def reorder_based_prediction_general_multiple_rec():
    """
    ° recommend the mostly reordered items to every user
    ° take a look on how well the recommendation performs by an increasing number of recommendations

    -> 1 item: ~0.98%
    -> 30 items: ~7.1%
    -> this goes up to about 16% for top 100 mostly recommended items
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder = total_reorders_count_for_every_item(train_df)
    reorders_train = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True)

    test_df = test_df[['itemID', 'order']]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        reorders_train_list = reorders_train['itemID'].tolist()[:num_recommendations]
        new_test_df = test_df

        # simply recommend the items with the most reorders to every user
        new_test_df['validation'] = new_test_df.apply(
            lambda row: True if row['itemID'] in reorders_train_list else False, axis=1)
        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'order']]
        error = (valid_groups['order'][0] / valid_groups['order'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["order"][1] / valid_groups["order"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    evaluation_result_df = pd.DataFrame(evaluation_result)
    evaluation_result_df.to_csv('reorder_based_prediction_general_multiple_rec.csv', sep='|')


def reorder_based_prediction_hybrid_user_spec_general_multiple_rec():
    """
    ° recommend the mostly reordered items specific for every user filled up to 30 with general recommended items

    -> 1 item from user spec, 29 from general: ~11%
    -> 29 from user spec, one from general: ~31.5%
    """
    MAX_NUMBER_OF_RECOMMENDATIONS = 30

    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder = total_reorders_count_for_every_item(train_df)
    reorder_per_user = reorder_count_same_user(train_df)
    reorders_train = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True)

    group = reorder_per_user.groupby(['userID'])
    keys = group.groups.keys()
    reorder_groups = [group.get_group(reorder_count).sort_values(['reorder'], ascending=False).reset_index(drop=True)
                      for reorder_count in keys]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, MAX_NUMBER_OF_RECOMMENDATIONS):
        user_spec_reorder = {'userID': [], 'validation': []}
        for user_id_df in reorder_groups:
            user_id_item_list = user_id_df['itemID'].tolist()
            if num_recommendations > len(user_id_item_list):
                items_from_general = reorders_train['itemID'].tolist()
                user_id_rec_list = user_id_item_list[:num_recommendations]
                # fill up the rest with items from general
                for general_item in items_from_general:
                    # append just general items that are not already in recommended
                    if general_item not in user_id_rec_list:
                        user_id_rec_list.append(general_item)
                    if len(user_id_rec_list) >= MAX_NUMBER_OF_RECOMMENDATIONS:
                        break
            test_item = test_df[test_df['userID'] == int(user_id_df['userID'][0])]['itemID'].tolist()[0]
            if test_item in user_id_rec_list:
                user_spec_reorder['validation'].append(True)
            else:
                user_spec_reorder['validation'].append(False)
            user_spec_reorder['userID'].append(user_id_df['userID'][0])
        new_test_df = pd.DataFrame(user_spec_reorder)

        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'userID']]
        error = (valid_groups['userID'][0] / valid_groups['userID'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["userID"][1] / valid_groups["userID"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    rec_df = pd.DataFrame(evaluation_result)
    rec_df.to_csv('reorder_based_prediction_hybrid_user_spec_general_multiple_rec.csv', sep='|')


def user_reorder_item_based_prediction_general_multiple_rec():
    """
    ° recommend items that get reordered from many users

    -> 1 item: ~0.98%
    -> 30 items: ~7.8%
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    item_reordered_by_users = count_users_that_reorder_item(train_df)
    reorders_train = item_reordered_by_users.sort_values(['reorder_count'], ascending=False).reset_index(drop=True)

    test_df = test_df[['itemID', 'order']]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        reorders_train_list = reorders_train['itemID'].tolist()[:num_recommendations]
        new_test_df = test_df

        # simply recommend the items that get reordered from the most users
        new_test_df['validation'] = new_test_df.apply(
            lambda row: True if row['itemID'] in reorders_train_list else False, axis=1)
        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'order']]
        error = (valid_groups['order'][0] / valid_groups['order'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["order"][1] / valid_groups["order"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    evaluation_result_df = pd.DataFrame(evaluation_result)
    evaluation_result_df.to_csv('user_reorder_item_based_prediction_general_multiple_rec.csv', sep='|')


def user_order_item_based_prediction_general_multiple_rec():
    """
    ° recommend items that get ordered from many users

    -> 1 item: ~0.98%
    -> 30 items: ~7.84%
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    item_ordered_by_users = count_users_that_order_item(train_df)
    orders_train = item_ordered_by_users.sort_values(['order_count'], ascending=False).reset_index(drop=True)

    test_df = test_df[['itemID', 'order']]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        reorders_train_list = orders_train['itemID'].tolist()[:num_recommendations]
        new_test_df = test_df

        # simply recommend the items that get ordered from the most users
        new_test_df['validation'] = new_test_df.apply(
            lambda row: True if row['itemID'] in reorders_train_list else False, axis=1)
        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'order']]
        error = (valid_groups['order'][0] / valid_groups['order'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["order"][1] / valid_groups["order"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    evaluation_result_df = pd.DataFrame(evaluation_result)
    evaluation_result_df.to_csv('user_order_item_based_prediction_general_multiple_rec.csv', sep='|')


def total_order_based_prediction_general_multiple_rec():
    """
    ° recommend items with highest total order count

    -> 1 item: ~0.98%
    -> 30 items: ~7.8%
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    total_item_order = count_users_that_order_item(train_df)
    orders_train = total_item_order.sort_values(['order_count'], ascending=False).reset_index(drop=True)

    test_df = test_df[['itemID', 'order']]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        reorders_train_list = orders_train['itemID'].tolist()[:num_recommendations]
        new_test_df = test_df

        # simply recommend the items with the most orders to every user
        new_test_df['validation'] = new_test_df.apply(
            lambda row: True if row['itemID'] in reorders_train_list else False, axis=1)
        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'order']]
        error = (valid_groups['order'][0] / valid_groups['order'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["order"][1] / valid_groups["order"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    evaluation_result_df = pd.DataFrame(evaluation_result)
    evaluation_result_df.to_csv('total_order_based_prediction_general_multiple_rec.csv', sep='|')


def reorder_based_prediction_hybrid_user_spec_general_multiple_feature_blacklist_rec():
    """
    ° recommend the mostly reordered items specific for every user filled up to 30 with general recommended items

    -> 1 item from user spec, 29 from general: ~11%
    -> 29 from user spec, one from general: ~31.5%
    """
    MAX_NUMBER_OF_RECOMMENDATIONS = 30

    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    train_feature_brand_values = pd.read_csv('../data_analysation/mapped_order_item_features_category_train_all_except_one.csv', delimiter='|')
    items = pd.read_csv('../resources/items.csv', delimiter='|')
    reorder = total_reorders_count_for_every_item(train_df)
    reorder_per_user = reorder_count_same_user(train_df)
    reorders_train = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True)

    # count how often a feature_1 value is ordered by all users
    feature_1_order_count = train_feature_brand_values.groupby(['feature_1']).count().reset_index()[['feature_1', 'order']].sort_values(['order'], ascending=False).reset_index(drop=True)
    # set the rarest feature_1 values on a blacklist to not recommend items with these feature_1 values
    feature_1_blacklist = feature_1_order_count['feature_1'][feature_1_order_count.shape[0] - 7:].tolist()

    group = reorder_per_user.groupby(['userID'])
    keys = group.groups.keys()
    reorder_groups = [group.get_group(reorder_count).sort_values(['reorder'], ascending=False).reset_index(drop=True)
                      for reorder_count in keys]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, MAX_NUMBER_OF_RECOMMENDATIONS):
        user_spec_reorder = {'userID': [], 'validation': []}
        for user_id_df in reorder_groups:
            user_id_item_list = user_id_df['itemID'].tolist()
            items_from_general = reorders_train['itemID'].tolist()
            user_id_rec_list = []
            for user_id_item in user_id_item_list:
                # recommend just these items who`s feature values arent on the blacklist
                if items[items['itemID'] == user_id_item].reset_index()['feature_1'][0] not in feature_1_blacklist:
                    user_id_rec_list.append(user_id_item)
                if len(user_id_rec_list) >= num_recommendations:
                    break
            # fill up the rest with items from general
            for general_item in items_from_general:
                # append just general items that are not already in recommended
                if general_item not in user_id_rec_list:
                    user_id_rec_list.append(general_item)
                if len(user_id_rec_list) >= MAX_NUMBER_OF_RECOMMENDATIONS:
                    break
            test_item = test_df[test_df['userID'] == int(user_id_df['userID'][0])]['itemID'].tolist()[0]
            if test_item in user_id_rec_list:
                user_spec_reorder['validation'].append(True)
            else:
                user_spec_reorder['validation'].append(False)
            user_spec_reorder['userID'].append(user_id_df['userID'][0])
        new_test_df = pd.DataFrame(user_spec_reorder)

        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'userID']]
        error = (valid_groups['userID'][0] / valid_groups['userID'][1]) - 1  # min 0, lower is better
        percentage_correct = valid_groups["userID"][1] / valid_groups["userID"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    rec_df = pd.DataFrame(evaluation_result)
    rec_df.to_csv('reorder_based_prediction_hybrid_user_spec_general_multiple_feature_blacklist_rec.csv', sep='|')


if __name__ == '__main__':
    # total_order_based_prediction_general_multiple_rec()
    # user_reorder_item_based_prediction_general_multiple_rec()
    # reorder_based_prediction_user_specific_multiple_rec()
    reorder_based_prediction_hybrid_user_spec_general_multiple_feature_blacklist_rec()
    # reorder_based_prediction_general_multiple_rec()

    # df_user_order = pd.read_csv('user_order_item_based_prediction_general_multiple_rec.csv', delimiter='|')
    # df_user_reorder = pd.read_csv('user_reorder_item_based_prediction_general_multiple_rec.csv', delimiter='|')
    # df_user_spec = pd.read_csv('reorder_based_prediction_user_specific_multiple_rec.csv', delimiter='|')
    # df_general = pd.read_csv('reorder_based_prediction_general_multiple_rec.csv', delimiter='|')
    # df_hybrid = pd.read_csv('reorder_based_prediction_hybrid_user_spec_general_multiple_rec.csv', delimiter='|')
    #
    # df_general['user_spec_correct_recommendations_[%]'] = df_user_spec['correct_recommendations_[%]']
    # df_general['general_correct_recommendations_[%]'] = df_general['correct_recommendations_[%]']
    # df_general['hybrid_correct_recommendations_[%]'] = df_hybrid['correct_recommendations_[%]']
    # df_general['user_order_correct_recommendations_[%]'] = df_user_order['correct_recommendations_[%]']
    # df_general['user_reorder_correct_recommendations_[%]'] = df_user_reorder['correct_recommendations_[%]']
    # df_general.plot(x='num_recommendations',
    #                 y=['user_spec_correct_recommendations_[%]',
    #                    'general_correct_recommendations_[%]',
    #                    'hybrid_correct_recommendations_[%]',
    #                    'user_order_correct_recommendations_[%]',
    #                    'user_reorder_correct_recommendations_[%]'])
    # plt.show()
    # pass

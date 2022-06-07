from data_analysation.item_user_analysations import reorder_from_same_user, total_order_count
import pandas as pd
import matplotlib.pyplot as plt


def reorder_based_prediction_general_single_rec():
    """
    ° recommend the mostly reordered item to every user

    -> one vs all: ~ 0.98% correct recommendations
    """
    orders = pd.read_csv('../resources/orders.csv', delimiter='|').sort_values(['userID'])

    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder, reorder_per_user = reorder_from_same_user(train_df)
    reorders_train_head = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True).head()
    # reorders_orders_head = reorder_from_same_user(orders).sort_values(['reorder_total'], ascending=False).reset_index(drop=True).head()

    # simply recommend the item with the most reorders to every user
    test_df['recommendation'] = test_df.apply(lambda row: reorders_train_head['itemID'][0], axis=1)
    # evaluate performance of the prediction
    test_df['validation'] = test_df.apply(lambda row: True if row['recommendation'] == row['itemID'] else False, axis=1)
    valid_groups = test_df.groupby(['validation']).count().reset_index()[['validation', 'recommendation']]
    error = (valid_groups['recommendation'][0] / valid_groups['recommendation'][1]) - 1     # min 0, lower is better
    print(f'correct recommendations: {valid_groups["recommendation"][1] / valid_groups["recommendation"][0] * 100}%')
    # -> about 0.977% correct recommendations


def reorder_based_prediction_user_specific_single_rec():
    """
    ° recommend the mostly reordered specific for every user
    -> one vs all: ~ 4.23% correct recommendations
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder, reorder_per_user = reorder_from_same_user(train_df)
    reorders_train_head = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True).head()

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
    reorder, reorder_per_user = reorder_from_same_user(train_df)

    group = reorder_per_user.groupby(['userID'])
    keys = group.groups.keys()
    reorder_groups = [group.get_group(reorder_count).sort_values(['reorder'], ascending=False).reset_index(drop=True) for reorder_count in keys]

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
        error = (valid_groups['userID'][0] / valid_groups['userID'][1]) - 1     # min 0, lower is better
        percentage_correct = valid_groups["userID"][1] / valid_groups["userID"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    rec_df = pd.DataFrame(evaluation_result)
    rec_df.to_csv('reorder_based_prediction_user_specific_multiple_rec.csv', sep='|')
    # rec_df[['correct_recommendations_[%]']].plot()
    # plt.show()
    pass


def reorder_based_prediction_general_multiple_rec():
    """
    ° recommend the mostly reordered items to every user
    ° take a look on how well the recommendation performs by an increasing number of recommendations

    -> this goes up to about 16% for top 100 mostly recommended items
    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder, reorder_per_user = reorder_from_same_user(train_df)
    reorders_train = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True)

    test_df = test_df[['itemID', 'order']]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        reorders_train_list = reorders_train['itemID'].tolist()[:num_recommendations]
        new_test_df = test_df

        # simply recommend the items with the most reorders to every user
        new_test_df['validation'] = new_test_df.apply(lambda row: True if row['itemID'] in reorders_train_list else False, axis=1)
        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'order']]
        error = (valid_groups['order'][0] / valid_groups['order'][1]) - 1     # min 0, lower is better
        percentage_correct = valid_groups["order"][1] / valid_groups["order"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    evaluation_result_df = pd.DataFrame(evaluation_result)
    evaluation_result_df.to_csv('reorder_based_prediction_general_multiple_rec.csv', sep='|')


def reorder_based_prediction_hybrid_user_spec_general_multiple_rec():
    """
    ° recommend the mostly reordered items specific for every user filled up to 30 with general recommended items

    """
    train_df = pd.read_csv('../resources/train_all_except_one.csv', delimiter='|')
    test_df = pd.read_csv('../resources/test_one_except_all.csv', delimiter='|')
    reorder, reorder_per_user = reorder_from_same_user(train_df)
    reorders_train = reorder.sort_values(['reorder_total'], ascending=False).reset_index(drop=True)

    group = reorder_per_user.groupby(['userID'])
    keys = group.groups.keys()
    reorder_groups = [group.get_group(reorder_count).sort_values(['reorder'], ascending=False).reset_index(drop=True) for reorder_count in keys]

    evaluation_result = {'num_recommendations': [], 'correct_recommendations_[%]': []}
    for num_recommendations in range(1, 30):
        user_spec_reorder = {'userID': [], 'validation': []}
        for user_id_df in reorder_groups:
            user_id_item_list = user_id_df['itemID'].tolist()
            if num_recommendations > len(user_id_item_list):
                items_from_general = reorders_train['itemID'].tolist()[:30 - len(user_id_item_list)]
                items_from_user_spec = user_id_item_list
                user_id_rec_list = items_from_user_spec + items_from_general
            else:
                items_from_general = reorders_train['itemID'].tolist()[:30 - num_recommendations]
                items_from_user_spec = user_id_item_list[:num_recommendations]
                user_id_rec_list = items_from_user_spec + items_from_general
            test_item = test_df[test_df['userID'] == int(user_id_df['userID'][0])]['itemID'].tolist()[0]
            if test_item in user_id_rec_list:
                user_spec_reorder['validation'].append(True)
            else:
                user_spec_reorder['validation'].append(False)
            user_spec_reorder['userID'].append(user_id_df['userID'][0])
        new_test_df = pd.DataFrame(user_spec_reorder)

        # evaluate performance of the prediction
        valid_groups = new_test_df.groupby(['validation']).count().reset_index()[['validation', 'userID']]
        error = (valid_groups['userID'][0] / valid_groups['userID'][1]) - 1     # min 0, lower is better
        percentage_correct = valid_groups["userID"][1] / valid_groups["userID"][0] * 100
        print(f'correct recommendations: {percentage_correct}%')
        evaluation_result['num_recommendations'].append(num_recommendations)
        evaluation_result['correct_recommendations_[%]'].append(percentage_correct)
    rec_df = pd.DataFrame(evaluation_result)
    rec_df.to_csv('reorder_based_prediction_hybrid_user_spec_general_multiple_rec.csv', sep='|')


if __name__ == '__main__':
    # reorder_based_prediction_user_specific_multiple_rec()
    # reorder_based_prediction_hybrid_user_spec_general_multiple_rec()

    # reorder_based_prediction_general_multiple_rec()
    df_user_spec = pd.read_csv('reorder_based_prediction_user_specific_multiple_rec.csv', delimiter='|')
    df_general = pd.read_csv('reorder_based_prediction_general_multiple_rec.csv', delimiter='|')
    df_hybrid = pd.read_csv('reorder_based_prediction_hybrid_user_spec_general_multiple_rec.csv', delimiter='|')

    df_general['user_spec_correct_recommendations_[%]'] = df_user_spec['correct_recommendations_[%]']
    df_general['general_correct_recommendations_[%]'] = df_general['correct_recommendations_[%]']
    df_general['hybrid_correct_recommendations_[%]'] = df_hybrid['correct_recommendations_[%]']
    df_general.plot(x='num_recommendations',
                    y=['user_spec_correct_recommendations_[%]',
                       'general_correct_recommendations_[%]',
                       'hybrid_correct_recommendations_[%]'])
    plt.show()
    pass

    # reorder_based_prediction_user_specific_multiple_rec()

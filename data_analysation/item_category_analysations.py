import pandas as pd
import matplotlib.pyplot as plt


def analyse_correlation_between_category_and_features():
    """checking correlation between item category and item features/brand.

    RESULTS:
    -> it seems like items in same category share about 75% of there feature values/brand.
    -> they pairwise difference in feature values/brand is only 0 to 2 features out of 6 (with brand)
    -> often one value from one feature is dominant over the other values in this feature
    (dominant means ~70% of the values in that features are the same)
    """
    # first do this analysation on one item
    items = pd.read_csv('../resources/items.csv', delimiter='|').sort_values(['itemID'])

    items['categories'] = items.apply(
        lambda row: row['categories'].strip('][').split(', ') if str(row['categories']) != 'nan' else [], axis=1)
    items = items.explode("categories")
    group = items.groupby(['categories'])
    keys = group.groups.keys()
    cat_groups = [group.get_group(cat) for cat in keys]
    # analyse results
    cat_1000 = cat_groups[2]        # select example category group -> here category 1000
    cat_1000_brand = pd.DataFrame(cat_1000.groupby(['brand'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_1 = pd.DataFrame(cat_1000.groupby(['feature_1'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_2 = pd.DataFrame(cat_1000.groupby(['feature_2'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_3 = pd.DataFrame(cat_1000.groupby(['feature_3'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_4 = pd.DataFrame(cat_1000.groupby(['feature_4'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    cat_1000_feature_5 = pd.DataFrame(cat_1000.groupby(['feature_5'])['categories'].count()).sort_values(['categories']).reset_index(drop=True)
    # plot feature values/brand against there occurrences
    cat_1000_brand.plot()
    cat_1000_feature_1.plot()
    cat_1000_feature_2.plot()
    cat_1000_feature_3.plot()
    cat_1000_feature_4.plot()
    cat_1000_feature_5.plot()
    plt.show()
    pass


#### meta model

# explanation
# In this competition I have designed  a hybrid recommendation system that combines XGBoost and SVD predictions using Ridge regression.
# It extracts user and business features from the data in multiple datasets, including user.json, business.json, checkin.json, photo.json, tip.json, and review_train.json.
# Features extracted include reviews count, average star rating, total compliments count, fans count, WiFi availability, total checkins count, photos count, statistical insights about tips, and text-based review characteristics like their length and perceived usefulness.
# In addition, interactive features as supplementary information include review count variability, engagement-visibility relationship variability, and elite friend ratio.
# The XGBoost model learns on this structured data, while SVD learns collaborative filtering signals arising from user-item interaction. 
# These predictions on a test dataset are merged using Ridge regression, which finds the weighting coefficients to use. 
# This hybrid approach attains RMSE reduction from about 0.99 to 0.9776 through leveraging both feature-driven and latent collaborative signals.

# RSME:0.9776998808908289
# Expected Time: 139.66 seconds


# Rating Prediction Error Breakdown:
# >=0 and <1 n = 102428
# >=1 and <2 n = 32635
# >=2 and <3 n = 6145
# >=3 and <4 n = 835
# >=4 n = 1


import os
import sys
import time
import json
import pandas as pd
import xgboost as xgb
from pyspark import SparkContext
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np
from surprise import SVD, Dataset, Reader
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split



os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'hybrid_svd_blend')
sc.setLogLevel("WARN")

def collect_rating_map(rdd, key1, key2, val_idx):
    return (rdd.map(lambda row: (row[key1], (row[key2], float(row[val_idx]))))
               .groupByKey()
               .mapValues(lambda vals: {k: v for k, v in vals})
               .collectAsMap())
def average_ratings(rdd, key_idx, val_idx):
    return (rdd.map(lambda row: (row[key_idx], float(row[val_idx])))
               .groupByKey()
               .mapValues(lambda x: sum(x) / len(x))
               .collectAsMap())

# function to extract features fromuser, business, photo, tips
def extract_features(df, b_map, u_map):
    user_keys = ['review_count', 'useful', 'fans', 'average_stars', 'funny', 'cool', 'elite_years', 'friend_count', 'compliment_sum']
    biz_keys = ['stars', 'review_count', 'RestaurantsPriceRange2', 'is_open', 'HasTV', 'RestaurantsTakeOut', 'OutdoorSeating', 'WiFi', 'checkin_count', 'photo_count', 'tip_count', 'avg_tip_length']
    features = defaultdict(list)

    for idx in range(len(df)):
        u = df.loc[idx, 'user_id']
        b = df.loc[idx, 'business_id']
        u_feats = u_map.get(u, [0] * len(user_keys))
        b_feats = b_map.get(b, [0] * len(biz_keys))

        for k, v in zip(user_keys, u_feats):
            features[f'user_{k}'].append(v)
        for k, v in zip(biz_keys, b_feats):
            features[f'biz_{k}'].append(v)

        features['user_avg_stars_diff'].append(u_feats[3] - b_feats[0])
        features['user_review_count_ratio'].append(u_feats[0] / (b_feats[1] + 1))
        features['biz_popularity'].append(b_feats[8] + b_feats[9] + b_feats[10])

        features['review_count_diff'].append(u_feats[0] - b_feats[1])
        features['compliment_to_popularity'].append(u_feats[8] / (b_feats[8] + b_feats[9] + b_feats[10] + 1))
        features['high_fan_high_star'].append(int(u_feats[2] > 5 and b_feats[0] > 4.0))
        features['elite_friend_star_diff'].append((u_feats[6] + u_feats[7]) - b_feats[0])
        features['biz_star_minus_user_avg'].append(b_feats[0] - u_feats[3])
        user_engagement = u_feats[0] + u_feats[1] + u_feats[8]
        biz_visibility = b_feats[1] + b_feats[8] + b_feats[9]
        features['engagement_vs_visibility'].append(user_engagement / (biz_visibility + 1))

    for col in features:
        df[col] = features[col]

    return df
if __name__ == '__main__':
    start_time = time.time()
    folder, test_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    train_df = pd.read_csv(os.path.join(folder, 'yelp_train.csv'))
    test_df = pd.read_csv(test_file)
    test_df['stars'] = test_df['stars'].astype(float)

    train_rdd = sc.textFile(os.path.join(folder, 'yelp_train.csv')).map(lambda x: x.split(',')).filter(lambda x: x[0] != 'user_id')
    user_to_items = collect_rating_map(train_rdd, 0, 1, 2)
    item_to_users = collect_rating_map(train_rdd, 1, 0, 2)
    user_avg = average_ratings(train_rdd, 0, 2)
    item_avg = average_ratings(train_rdd, 1, 2)

    business_json = sc.textFile(os.path.join(folder, 'business.json')).map(json.loads)
    user_json = sc.textFile(os.path.join(folder, 'user.json')).map(json.loads)
    checkin_json = sc.textFile(os.path.join(folder, 'checkin.json')).map(json.loads)
    photo_json = sc.textFile(os.path.join(folder, 'photo.json')).map(json.loads)
    tip_json = sc.textFile(os.path.join(folder, 'tip.json')).map(json.loads)

    review_json = sc.textFile(os.path.join(folder, 'review_train.json')).map(json.loads)
    # Features extracted from reviews
    review_rdd = review_json.map(lambda x: (
        (x['user_id'], x['business_id']),
        {
            'review_length': len(x['text']),
            'useful_review': x.get('useful', 0),
            'funny_review': x.get('funny', 0),
            'cool_review': x.get('cool', 0)
        }
    ))

    review_df = pd.DataFrame([
        {'user_id': uid, 'business_id': bid, **features}
        for (uid, bid), features in review_rdd.collect()
    ])

    checkin_map = checkin_json.map(lambda x: (x['business_id'], sum(x.get('time', {}).values()))).collectAsMap()
    photo_map = photo_json.map(lambda x: (x['business_id'], 1)).groupByKey().mapValues(len).collectAsMap()
    tip_map = tip_json.map(lambda x: (x['business_id'], (1, len(x['text']))))
    tip_agg = tip_map.aggregateByKey((0, 0), lambda acc, val: (acc[0] + val[0], acc[1] + val[1]), lambda a, b: (a[0] + b[0], a[1] + b[1]))
    tip_final = tip_agg.mapValues(lambda x: (x[0], x[1]/x[0] if x[0] > 0 else 0)).collectAsMap()

    b_map = business_json.map(lambda x: (
        x['business_id'],
        [
            x.get('stars', 0),
            x.get('review_count', 0),
            int(x.get('attributes', {}).get('RestaurantsPriceRange2', 0)) if x.get('attributes') else 0,
            x.get('is_open', 0),
            int((x.get('attributes') or {}).get('HasTV', 'False') == 'True'),
            int((x.get('attributes') or {}).get('RestaurantsTakeOut', 'False') == 'True'),
            int((x.get('attributes') or {}).get('OutdoorSeating', 'False') == 'True'),
            int((x.get('attributes') or {}).get('WiFi', 'no') not in ['no', 'None', None]),
            checkin_map.get(x['business_id'], 0),
            photo_map.get(x['business_id'], 0),
            tip_final.get(x['business_id'], (0, 0))[0],
            tip_final.get(x['business_id'], (0, 0))[1]
        ]
    )).collectAsMap()

    u_map = user_json.map(lambda x: (
        x['user_id'],
        [
            x.get('review_count', 0),
            x.get('useful', 0),
            x.get('fans', 0),
            x.get('average_stars', 0),
            x.get('funny', 0),
            x.get('cool', 0),
            len(x.get('elite', '').split(',')) if x.get('elite') and x['elite'] != 'None' else 0,
            len(x.get('friends', '').split(',')) if x.get('friends') and x['friends'] != 'None' else 0,
            sum(x.get(k, 0) for k in x if k.startswith("compliment_"))
        ]
    )).collectAsMap()

    train_df = extract_features(train_df, b_map, u_map)
    test_df = extract_features(test_df, b_map, u_map)

    train_df.to_csv('train_data_interactions.csv', index=False)
    test_df.to_csv('test_data_interactions.csv', index=False)

    X_train = train_df.drop(['user_id', 'business_id', 'stars'], axis=1).apply(pd.to_numeric)
    y_train = train_df['stars'].astype(float)
    X_test = test_df.drop(['user_id', 'business_id', 'stars'], axis=1).apply(pd.to_numeric)

    # Split training data for meta-model validation
    train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    #  Train on XGBoost
    xgb_model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=2.0,
        random_state=42
    )
    xgb_model.fit(train_X, train_y)
    xgb_val_preds = xgb_model.predict(val_X)
    xgb_preds = xgb_model.predict(X_test)

    # Training on SVD
    svd_df_train = train_df.iloc[train_X.index][['user_id', 'business_id', 'stars']]
    reader = Reader(rating_scale=(0.0, 5.0))
    svd_data_small = Dataset.load_from_df(svd_df_train, reader)
    svd_trainset = svd_data_small.build_full_trainset()
    svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.1)
    svd_model.fit(svd_trainset)

    val_user_ids = train_df.iloc[val_X.index]['user_id'].values
    val_biz_ids = train_df.iloc[val_X.index]['business_id'].values
    svd_val_preds = [svd_model.predict(uid, bid).est for uid, bid in zip(val_user_ids, val_biz_ids)]

    # using ridge as meta model
    meta_X_val = np.vstack([xgb_val_preds, svd_val_preds]).T
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(meta_X_val, val_y)

    test_user_ids = test_df['user_id'].values
    test_biz_ids = test_df['business_id'].values
    svd_preds = [svd_model.predict(uid, bid).est for uid, bid in zip(test_user_ids, test_biz_ids)]

    #Final prediction prediction
    meta_X_test = np.vstack([xgb_preds, svd_preds]).T
    final_test_preds = ridge_model.predict(meta_X_test)

    # outcome stored as csv file
    with open(output_file, 'w') as f:
        f.write("user_id,business_id,prediction\n")
        for idx, row in test_df.iterrows():
            u, b = row['user_id'], row['business_id']
            pred = final_test_preds[idx]
            f.write(f"{u},{b},{pred}\n")

    # rsme calculated
    gt = test_df['stars'].tolist()
    rmse = np.sqrt(mean_squared_error(gt, final_test_preds))
    print("RMSE:", rmse)
    print("Duration:", time.time() - start_time)

    # # Error analysis: Absolute differences
    # abs_diff = np.abs(np.array(gt) - np.array(final_test_preds))

    # # Count distribution of errors
    # bins = [0, 1, 2, 3, 4, np.inf]
    # labels = ['>=0 and <1', '>=1 and <2', '>=2 and <3', '>=3 and <4', '>=4']
    # error_distribution = pd.cut(abs_diff, bins=bins, labels=labels).value_counts().sort_index()

    # print("\n🔍 Rating Prediction Error Breakdown:")
    # for label in labels:
    #     print(f"{label} n = {error_distribution[label]}")

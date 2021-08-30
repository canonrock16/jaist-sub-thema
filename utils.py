import numpy as np
import pandas as pd

# コサイン類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ユーザー×アイテムの評価行列を作成
def get_rate_matrix():
    # データインポート
#     data = pd.read_table("ml-100k/u.data", header=None, usecols=[0, 1, 2], names=["user_id", "item_id", "rating"])
    data = pd.read_table("ml-1m/ratings.dat", sep='::',header=None, usecols=[0, 1, 2], names=["user_id", "item_id", "rating"])

    # ndarrayの行番号とユーザーID,列番号とアイテムIDを対応付ける辞書を作成
    user_id2row_num = {}
    row_num2user_id = {}
    for i, user_id in enumerate(set(list(data["user_id"]))):
        user_id2row_num[user_id] = i
        row_num2user_id[i] = user_id

    item_id2column_num = {}
    column_num2item_id = {}
    for i, item_id in enumerate(set(list(data["item_id"]))):
        item_id2column_num[item_id] = i
        column_num2item_id[i] = item_id

    # ユーザー×アイテムの評価行列を作成
    rate_matrix = np.zeros((len(user_id2row_num), len(item_id2column_num)))
    # for row in tqdm(data.itertuples(), total=data.shape[0]):
    for row in data.itertuples():
        rate_matrix[user_id2row_num[row.user_id], item_id2column_num[row.item_id]] = row.rating

    return rate_matrix, user_id2row_num, row_num2user_id, item_id2column_num, column_num2item_id

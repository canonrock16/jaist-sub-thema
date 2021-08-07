import itertools

import numpy as np
import pandas as pd
from mpi4py import MPI
from surprise import Dataset, KNNBasic

# 設定値
core_num = 8
k = 5
predict_user_id = 2


comm = MPI.COMM_WORLD  # 並列処理開始

size = comm.Get_size()  # 並列処理に使用できるプロセッサ数
rank = comm.Get_rank()  # 各プロセッサのIDのようなもの

# コサイン類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ユーザー×アイテムの評価行列を作成
def get_rate_matrix():
    # データインポート
    data = pd.read_table("ml-100k/u.data", header=None, usecols=[0, 1, 2], names=["user_id", "item_id", "rating"])

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


# プロセス0が担当領域の分割と各プロセスへの送信、類似度計算結果の取得を行い、結果を収集してpredictする
if rank == 0:
    rate_matrix, user_id2row_num, row_num2user_id, item_id2column_num, column_num2item_id = get_rate_matrix()

    # ユーザー×ユーザーの組み合わせを列挙
    row_list = [i for i in range(len(row_num2user_id))]
    comb = [c for c in itertools.combinations(row_list, 2)]

    # それぞれのコアにできるだけ均等になるように担当の組み合わせを配布
    # TODO:実行時に指定されたコア数で分割するように変更
    comb_list = np.array_split(comb, core_num - 1)
    for i in range(core_num - 1):
        comm.send(rate_matrix, dest=i + 1,tag=0)
        comm.send(comb_list[i], dest=i + 1,tag=1)

    # 類似度計算結果を格納するユーザー×ユーザー行列を作成
    sim_matrix = np.zeros((len(user_id2row_num), len(user_id2row_num)))
    for i in range(core_num - 1):
        sim_list = comm.recv(source=i + 1)
        for s in sim_list:
            sim_matrix[s["comb"][0], s["comb"][1]] = s["sim"]

    # 類似度が上位k件のユーザーIDリストを作成
    predict_row_num = user_id2row_num[predict_user_id]
    top_k_rows = np.argsort(sim_matrix[predict_row_num])[::-1][:k]

    topk_mean_ratings = np.mean(rate_matrix[top_k_rows, :], axis=0)
    for i in np.argsort(topk_mean_ratings)[::-1]:
        if rate_matrix[predict_row_num, i] == 0:
            print('---result---')
            print(f"itemid:{column_num2item_id[i]},predicted_score:{topk_mean_ratings[i]}")
            break

else:
    # 各プロセスは担当の組み合わせの類似度を計算し、プロセス0へ返信
    rate_matrix = comm.recv(source=0,tag=0)
    calc_combs = comm.recv(source=0,tag=1)

    # コサイン類似度を計算
    sim_list = []
    for c in calc_combs:
        sim_dict = {}
        sim_dict["comb"] = c
        sim_dict["sim"] = cos_sim(rate_matrix[c[0]], rate_matrix[c[1]])
        sim_list.append(sim_dict)

    comm.send(sim_list, dest=0)
    print("Hello world {0} / {1}".format(rank, size))
    print(f'#{rank}/{size} complete!')
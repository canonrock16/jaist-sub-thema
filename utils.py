import itertools

import numpy as np
import pandas as pd
from mpi4py import MPI


class MovieLensDataSet:
    def __init__(self) -> None:
        # self.data = pd.read_table("ml-1m/ratings.dat", sep="::", header=None, usecols=[0, 1, 2], names=["user_id", "item_id", "rating"])
        self.data = pd.read_table("ml-100k/u.data", header=None, usecols=[0, 1, 2], names=["user_id", "item_id", "rating"])

        self.user_id2row_num = {}
        self.row_num2user_id = {}
        self.item_id2column_num = {}
        self.column_num2item_id = {}
        self.rate_matrix = ""

    # ユーザー×アイテムの評価行列を作成
    def get_rate_matrix(self):
        # ndarrayの行番号とユーザーID,列番号とアイテムIDを対応付ける辞書を作成
        for i, user_id in enumerate(set(list(self.data["user_id"]))):
            self.user_id2row_num[user_id] = i
            self.row_num2user_id[i] = user_id

        for i, item_id in enumerate(set(list(self.data["item_id"]))):
            self.item_id2column_num[item_id] = i
            self.column_num2item_id[i] = item_id

        # ユーザー×アイテムの評価行列を作成
        self.rate_matrix = np.zeros((len(self.user_id2row_num), len(self.item_id2column_num)))
        # for row in tqdm(data.itertuples(), total=data.shape[0]):
        for row in self.data.itertuples():
            self.rate_matrix[self.user_id2row_num[row.user_id], self.item_id2column_num[row.item_id]] = row.rating

        return self.rate_matrix

    def get_combination_list(self, size):
        # ユーザー×ユーザーの組み合わせを列挙
        row_list = [i for i in range(len(self.row_num2user_id))]
        comb = [c for c in itertools.combinations(row_list, 2)]
        comb_list = np.array_split(comb, size)

        return comb_list

    def get_topk_list(self, predict_user_id, sim_matrix, k):
        # 類似度が上位k件のユーザーIDリストを作成
        predict_row_num = self.user_id2row_num[predict_user_id]
        top_k_rows = np.argsort(sim_matrix[predict_row_num])[::-1][:k]

        topk_mean_ratings = np.mean(self.rate_matrix[top_k_rows, :], axis=0)
        for i in np.argsort(topk_mean_ratings)[::-1]:
            if self.rate_matrix[predict_row_num, i] == 0:
                print("---result---")
                print(f"itemid:{self.column_num2item_id[i]},predicted_score:{topk_mean_ratings[i]}")
                break


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc_cos_sim(calc_combs, rate_matrix, rank):
    # コサイン類似度を計算
    sim_list = []
    for c in calc_combs:
        sim_dict = {}
        sim_dict["comb"] = c
        sim_dict["sim"] = cos_sim(rate_matrix[c[0]], rate_matrix[c[1]])
        sim_list.append(sim_dict)

    return sim_list

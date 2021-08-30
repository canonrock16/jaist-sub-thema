import itertools
import numpy as np
import time
from utils import cos_sim,get_rate_matrix
from tqdm import tqdm

start = time.time()

# 設定値
core_num = 8
k = 5
predict_user_id = 2

# プロセス0が担当領域の分割と各プロセスへの送信、類似度計算結果の取得を行い、結果を収集してpredictする
rate_matrix, user_id2row_num, row_num2user_id, item_id2column_num, column_num2item_id = get_rate_matrix()

# ユーザー×ユーザーの組み合わせを列挙
row_list = [i for i in range(len(row_num2user_id))]
comb = [c for c in itertools.combinations(row_list, 2)]

# コサイン類似度を計算
sim_list = []
for c in tqdm(comb):
    sim_dict = {}
    sim_dict["comb"] = c
    sim_dict["sim"] = cos_sim(rate_matrix[c[0]], rate_matrix[c[1]])
    sim_list.append(sim_dict)

# 類似度計算結果を格納するユーザー×ユーザー行列を作成
sim_matrix = np.zeros((len(user_id2row_num), len(user_id2row_num)))
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


elapsed_time = time.time() - start
print(f"elapsed_time:{elapsed_time}[sec]")

import fire
import numpy as np
from mpi4py import MPI

from utils import MovieLensDataSet, calc_cos_sim


def main(predict_user_id: int = 2, k: int = 5) -> None:
    start = MPI.Wtime()

    comm = MPI.COMM_WORLD  # 並列処理開始
    size = comm.Get_size()  # 並列処理に使用できるプロセッサ数。mpiexecコマンドで指定した数が設定される
    rank = comm.Get_rank()  # 各プロセッサのIDのようなもの

    # プロセス0が担当領域の分割と各プロセスへの送信、類似度計算結果の取得を行い、結果を収集してpredictする
    if rank == 0:
        start_make_comb_list = MPI.Wtime()

        dataset = MovieLensDataSet()
        rate_matrix = dataset.get_rate_matrix()
        comb_list = dataset.get_combination_list(size=size)

        print("評価行列作成にかかった時間,", MPI.Wtime() - start_make_comb_list)

        # それぞれのコアにできるだけ均等になるように担当の組み合わせを配布
        start_distribution = MPI.Wtime()
        for i in range(1, size):
            comm.send(rate_matrix, dest=i, tag=0)
            comm.send(comb_list, dest=i, tag=1)
        print("各コアへの組み合わせ配布にかかった時間,", MPI.Wtime() - start_distribution)

        start_calc_sim = MPI.Wtime()
        sim_list = calc_cos_sim(calc_combs=comb_list[rank], rate_matrix=rate_matrix, rank=rank)
        print(f"プロセス{rank}がコサイン類似度を計算するのにかかった時間,", MPI.Wtime() - start_calc_sim)

        start_gather = MPI.Wtime()
        results = []
        results.append(sim_list)
        for i in range(1, size):
            results.append(comm.recv(source=i))
        print(f"結果を収集するのにかかった時間,", MPI.Wtime() - start_gather)

        start_housing = MPI.Wtime()
        # 類似度計算結果を格納するユーザー×ユーザー行列を作成
        sim_matrix = np.zeros((len(dataset.user_id2row_num), len(dataset.user_id2row_num)), dtype="int32")

        for result in results:
            for s in result:
                sim_matrix[s["comb"][0], s["comb"][1]] = s["sim"]
        print(f"全ての結果を類似度行列に格納するのにかかった時間,", MPI.Wtime() - start_housing)

        start_topk = MPI.Wtime()
        dataset.get_topk_list(predict_user_id=predict_user_id, sim_matrix=sim_matrix, k=k)
        print(f"topkリストを計算するのにかかった時間,", MPI.Wtime() - start_topk)

        elapsed_time = MPI.Wtime() - start
        print(f"elapsed_time,{elapsed_time}")

    if rank != 0:
        rate_matrix = comm.recv(source=0, tag=0)
        comb_list = comm.recv(source=0, tag=1)

        start_calc_sim = MPI.Wtime()
        sim_list = calc_cos_sim(calc_combs=comb_list[rank], rate_matrix=rate_matrix, rank=rank)
        print(f"プロセス{rank}がコサイン類似度を計算するのにかかった時間,", MPI.Wtime() - start_calc_sim)

        comm.send(sim_list, dest=0)


if __name__ == "__main__":
    fire.Fire(main)

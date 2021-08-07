from mpi4py import MPI

comm = MPI.COMM_WORLD  # 並列処理開始です！

size = comm.Get_size() # 並列処理に使用できるプロセッサ数
rank = comm.Get_rank() # 各プロセッサのIDのようなもの

iter_num=100000000
result = 0
for i in range(iter_num):
    result = result + i

print("Hello world {0} / {1}".format(rank, size))
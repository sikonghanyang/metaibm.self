import test1  # 你的自定义模型
from test1 import mkdir_if_not_exist
from mpi4py import MPI  # MPI并行库
import numpy as np


# 初始化MPI环境
comm = MPI.COMM_WORLD  # 获取默认通信器
size = comm.Get_size()  # 获取总进程数
rank = comm.Get_rank()  # 获取当前进程的编号（0, 1, 2, ..., size-1）

rep_paras = np.arange(0,3)
baseline_jce = np.array([0.51,0.356,0.01005])
decay_jce = np.array([5,10,15])
all_jobs_parameters = [
    (k ,i, j)
    for k in rep_paras
    for i in baseline_jce
    for j in decay_jce
]
rep, baseline_jce, decay_jce= all_jobs_parameters[rank]

goal_path = mkdir_if_not_exist(
    rep, baseline_jce,  decay_jce)

if rank < len(all_jobs_parameters):
    test1.main(
    rep, baseline_jce,  decay_jce, goal_path
)# 执行任务
else :
    print(f"Rank {rank} has no work to do")
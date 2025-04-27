'''import test1  
from test1 import mkdir_if_not_exist
from mpi4py import MPI  # MPI并行库
import numpy as np


# 初始化MPI环境
comm = MPI.COMM_WORLD  # 获取默认通信器
size = comm.Get_size()  # 获取总进程数
rank = comm.Get_rank()  # 获取当前进程的编号（0, 1, 2, ..., size-1）

rep_paras = np.arange(0,3)
baseline_jce = np.array([0.51,0.356,0.01005])
decay_jce = np.array([1,5,10,15])
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
    print(f"Rank {rank} has no work to do")'''

import test1  
from test1 import mkdir_if_not_exist
from mpi4py import MPI
import numpy as np

# 初始化MPI环境
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# 生成所有任务参数
rep_paras = np.arange(0, 3)
baseline_jce = np.array([0.51, 0.356, 0.01005])
decay_jce = np.array([1, 5, 10, 15])
all_jobs_parameters = [
    (k, i, j)
    for k in rep_paras
    for i in baseline_jce
    for j in decay_jce
]

if rank == 0:
    # 主进程负责任务分发
    next_job = 0
    total_jobs = len(all_jobs_parameters)
    
    # 先给每个工作进程分配初始任务
    for worker in range(1, size):
        if next_job < total_jobs:
            comm.send(all_jobs_parameters[next_job], dest=worker)
            next_job += 1
        else:
            comm.send(None, dest=worker)  # 没有更多任务
    
    # 主进程处理自己的任务
    while next_job < total_jobs:
        rep, baseline_jce, decay_jce = all_jobs_parameters[next_job]
        goal_path = mkdir_if_not_exist(rep, baseline_jce, decay_jce)
        test1.main(rep, baseline_jce, decay_jce, goal_path)
        next_job += 1
    
    # 接收工作进程的完成信号并继续分配任务
    active_workers = size - 1
    while active_workers > 0:
        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, status=status)  # 接收工作进程的请求
        worker_rank = status.source
        
        if next_job < total_jobs:
            comm.send(all_jobs_parameters[next_job], dest=worker_rank)
            next_job += 1
        else:
            comm.send(None, dest=worker_rank)  # 通知工作进程结束
            active_workers -= 1
    
    print("所有任务已完成！")
else:
    # 工作进程循环请求并处理任务
    while True:
        # 请求任务
        comm.send("ready", dest=0)
        task = comm.recv(source=0)
        
        if task is None:  # 收到结束信号
            break
        
        rep, baseline_jce, decay_jce = task
        goal_path = mkdir_if_not_exist(rep, baseline_jce, decay_jce)
        test1.main(rep, baseline_jce, decay_jce, goal_path)
        print(f"Rank {rank}: 正在处理 rep={rep}, baseline={baseline_jce}, decay={decay_jce}")
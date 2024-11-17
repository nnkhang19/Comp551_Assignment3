import os
import numpy as np 

root = "logs/results"
# files = ["/home/mila/k/khang.ngo/Comp551/logs/results/l1_reg_1e2.npz", 
#          "/home/mila/k/khang.ngo/Comp551/logs/results/l1_reg_1e3.npz", 
#          "/home/mila/k/khang.ngo/Comp551/logs/results/l1_reg_1e4.npz",
#          "/home/mila/k/khang.ngo/Comp551/logs/results/l1_reg_1e5.npz"]
files = ["/home/mila/k/khang.ngo/Comp551/logs/results/unormalized.npz"]

# files = ["/home/mila/k/khang.ngo/Comp551/logs/results/task2_leaky_relu.npz", 
#          "/home/mila/k/khang.ngo/Comp551/logs/results/task3_l1.npz", 
#          "/home/mila/k/khang.ngo/Comp551/logs/results/task2_tanh.npz", 
#          "/home/mila/k/khang.ngo/Comp551/logs/results/task1_net1.npz", 
#          '/home/mila/k/khang.ngo/Comp551/logs/results/task1_net2.npz', 
#          '/home/mila/k/khang.ngo/Comp551/logs/results/task1_net3.npz', 
#          '/home/mila/k/khang.ngo/Comp551/logs/results/task3_l2.npz']


### plot results for unnormalized and normalized data ####
files = [
    "/home/mila/k/khang.ngo/Comp551/logs/results/unnormalized.npz", 
    "/home/mila/k/khang.ngo/Comp551/logs/results/normalized.npz"
]

# for file_ in files:
#     results = np.load(file_, allow_pickle=True)
#     args = results['config']
#     print(args)
#     for key in results.keys():
#         if key == "loss":
#             print(results[key][-1])
#         elif key == "test_acc" or key == "test_auc":
#             print(key, results[key])
#         else:
#             continue
    
#     print("*" * 10)

import torch
import pickle

weights_file = 'results/i2sb_128_bak/latest.pt'
config_file  = 'results/i2sb_128_bak/options.pkl' 

weights = torch.load(weights_file)

iters = weights['sched']['last_epoch']
print(iters)
#with open(config_file, 'rb') as f:
#    pkl = pickle.load(f)

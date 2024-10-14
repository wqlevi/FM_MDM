import torch
from i2sb.torchcfm.models.unet import nestedUnet

net = nestedUnet('cpu', min_size=64, training_mode = 'MDM256x256')

with open("256x256_MDM_FM_netarch.log",'w') as f:
    print(net, file=f)

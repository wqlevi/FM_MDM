# - [ ] ckpt not working with eval
# - [ ] weight in ckpt changed
import torch
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchdyn.core import NeuralODE
from prefetch_generator import BackgroundGenerator
from dataset.LMDB2ImageFolder import Dset
from i2sb.torchcfm.models.unet import UNetModel
from i2sb.torchcfm.conditional_flow_matching import *
from build_mask import build_inpaint_center
from PIL import Image
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

data_path = Dset(Path("/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/", "train", "subclass_train"), 256)
dataloader = DataLoaderX(data_path, batch_size=1, drop_last=True)
rank = 0
clean_img, y = next(iter(dataloader))
clean_img = clean_img.to(rank)

corrupt_method = build_inpaint_center(256, rank)
corrupt_img, mask = corrupt_method(clean_img)

model = UNetModel(dim=(3, 256, 256), num_channels=128, num_res_blocks=2).to(rank) 
ckpt = torch.load("./results/debug_runtime/latest.pt")
new_dict = {}
for k,v in ckpt['net'].items():
    new_dict[k.replace("module.","")] = v



model.load_state_dict(new_dict)
node = NeuralODE(model, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
#FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

traj = node.trajectory(corrupt_img, t_span=torch.linspace(0,1,2).to(rank))
im = Image.fromarray(traj[-1].detach().cpu().numpy())
im.save("test.png")


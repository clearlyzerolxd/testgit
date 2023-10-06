from Conv_tools import u2net_full
import torch
import numpy
import einops
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
pth_path = ""
example_pth = ""
class Get_tager_sample(Dataset):

    def __init__(self,path):
        self.img_path = os.listdir(path)
        self.path = path
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        radar = numpy.load(os.path.join(self.path,img_name))
        # Mytrainsform(radar)
        radar =torch.from_numpy(radar)

        tagert = (radar[0:4,:,:,:]- 0.4202) / 0.8913
        sample = (radar[4:10,:,:,:]- 0.4202) / 0.8913
        return tagert,sample

    def __len__(self):
        return len(self.img_path)


data = Get_tager_sample("")
val_t = DataLoader(data, batch_size=2,shuffle=False,drop_last=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = u2net_full(device='cuda:0').to(device)
net.load_state_dict(torch.load(pth_path))
net.eval()
x,y = next(iter(val_t))
with torch.no_grad():
    out = net(x)
    
    





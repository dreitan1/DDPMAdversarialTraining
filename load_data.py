import torch
import torchvision

import numpy as np

from tqdm import tqdm
import os

from torch.utils.data import TensorDataset, DataLoader, IterableDataset


class Dataset(IterableDataset):
    '''
        Dataset has data of the form (param, clean_img, adv_img)
    '''
    def __init__(self, PATH):
        param_dir = f'{PATH}/params'
        clean_dir = f'{PATH}/clean_imgs'
        adv_dir = f'{PATH}/adv_imgs'

        data = [[], [], []]

        params = [p for p in os.scandir(param_dir) if p.is_file()]
        for f in tqdm(params, desc="Loading train data"):  
            param_name = f.name.replace('.txt', '')

            clean_ims = [c for c in os.scandir(f'{clean_dir}/{param_name}') if c.is_file()]
            clean_ims.sort(key=lambda c:c.name)
            clean_ims = [c.path for c in clean_ims]

            adv_ims = [a for a in os.scandir(f'{adv_dir}/{param_name}') if a.is_file()]
            adv_ims.sort(key=lambda a:a.name)
            adv_ims = [a.path for a in adv_ims]

            with open(f.path, 'r') as p:
                param = p.read().split('\n')[:-1]
                param = np.asarray(param).astype(np.float32)
                param = torch.Tensor(param).to(torch.float32)

            for c, a in zip(clean_ims, adv_ims):
                im = torchvision.io.read_image(c).to(torch.float32) / 255
                adv = torchvision.io.read_image(a).to(torch.float32) / 255

                data[0].append(param)
                data[1].append(im)
                data[2].append(adv)

        self.params, self.ims, self.advs = data

    
    def __iter__(self):
        for p, i, a in zip(self.params, self.ims, self.advs):
            yield p, i, a


    def __len__(self):
        return len(self.params)


if __name__ == "__main__":
    data = Dataset('./train_data')

    batch_size = 16
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=2)

    for step, batch in enumerate(dataloader):
        p, i, a = batch
        print(step, p.shape, i.shape, a.shape)

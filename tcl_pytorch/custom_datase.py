from subfunc.generate_artificial_data import generate_artificial_data
from subfunc.preprocessing import pca
import torch.utils.data as data
import torch
class SimulatedDataset(data.Dataset):
    def __init__(self, num_comp, 
                 num_segment, num_segmentdata,num_layer,random_seed):
        # Generate sensor signal --------------------------------------
        self.sensor, self.source, self.label = generate_artificial_data(num_comp=num_comp,
                                                        num_segment=num_segment,
                                                        num_segmentdata=num_segmentdata,
                                                        num_layer=num_layer,
                                                        random_seed=random_seed)


        # Preprocessing -----------------------------------------------
        self.sensor, self.pca_parm = pca(self.sensor, num_comp=num_comp)
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.sensor[:,idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.label[idx], dtype=torch.long)
        return data_tensor,label_tensor
    
    def __getinputsize__(self):
        return self.sensor.shape[0]
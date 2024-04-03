import face_models
import torch
import torch.nn as nn
# import torch.nn.functional as F

class Architecture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = face_models._resnet(face_models.SISTCM, [2, 2, 2, 2])
        self.lstm = face_models.BiLSTM(config["localSpatTemprepresentation"], config["clipLevelRepresentation"], config["BiLSTM_hidden_size"], config["BiLSTM_num_layers"], config["out_classes"])
        
    def forward(self, x):
        # x -> Batch x Clips x F x C x H x W
        batch_size, num_clips, F, C, H, W = x.size()
        batches1 = []
        batches2 = []
        for batch_idx in range(batch_size):
            values1 = []
            values2 = []
            for clip_idx in range(num_clips):
                value = x[batch_idx, clip_idx, :, :, :, :]
                v1, v2 = self.resnet(value)
                values1.append(v1)
                values2.append(v2)
            values1 = torch.stack(values1)
            values2 = torch.stack(values2)
            batches1.append(values1)        
            batches2.append(values2)
        batches1 = torch.stack(batches1)
        batches2 = torch.stack(batches2)    
        
        out = self.lstm(batches1, batches2) # Batch x out_classes
        return out
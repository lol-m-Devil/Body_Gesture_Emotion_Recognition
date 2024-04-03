import torch
import os
from torch.utils.data import Dataset
import poses 

class VideoDataset(Dataset):
    def __init__(self, path, config):
        self.data_path = os.path.join(path)  # training or validation!      
        self.frames = config["frames"]
        self.videos = []
        self.labels = []
        videos = os.listdir(self.data_path)
        for video in videos:
            self.videos.append(os.path.join(path, video))
            label = VideoDataset.extractLabel(video)
            self.labels.append(label)
        
    @staticmethod    
    def extractLabel(filename):     
        identifiers = filename.split('-')
        return int(identifiers[2])
        
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # I need to convert video to  first, 
        
        keyframes = poses.bodyJoints(self.videos[idx])
        # random x 33 x 2
        length_keyframes = keyframes.shape[0]
        sampling_index = [x/self.frames for x in range(self.frames)]
        final_index = [int(i*length_keyframes) for i in sampling_index]
        
        sampled_frames = []
        for index in final_index:
            sampled_frames.append(keyframes[index])
        
        sampled_frames = torch.stack(sampled_frames)    
                
        return [sampled_frames, self.labels[idx]]
            

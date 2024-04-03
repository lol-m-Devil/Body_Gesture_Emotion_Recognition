import torch
import os
from torch.utils.data import Dataset
import face_detector
  
class VideoDataset(Dataset):
    def __init__(self, path, config):
        self.data_path = os.path.join(path)  # training or validation!      
        self.framesPerClip = config["framesPerClip"]
        self.clips = config["clips"]
        
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
        NumFrames = self.framesPerClip * self.clips
        keyframes = face_detector.face_detector(self.videos[idx])

        length_keyframes = len(keyframes)
        sampling_index = [x/NumFrames for x in range(NumFrames)]
        final_index = [int(i*length_keyframes) for i in sampling_index]
        
        sampled_frames = []
        for index in final_index:
            sampled_frames.append(keyframes[index])
        
        sampled_clips = []
        
        for i in range(self.clips):
            sampled_clips.append(torch.stack(sampled_frames[i*self.framesPerClip:(i+1)*self.framesPerClip], dim =0))
        
        sampled_clips = torch.stack(sampled_clips, dim = 0)    
        # print(sampled_clips.shape)
        
        return [sampled_clips, self.labels[idx]]
            
            
        



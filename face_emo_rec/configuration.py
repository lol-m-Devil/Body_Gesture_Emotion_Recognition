from pathlib import Path
import os

def get_config():
    return {
        "batch_size": 4,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "num_epochs": 3,
        "lr": 0.005,
        "step_size": 50,  
        "gamma": 0.5,
        "train_test_split": 0.9,
        "model_folder": "face_emo_rec/weights",
        "model_basename": "EmoDet_",
        "experiment_name": "face_emo_rec/experiment_1",
        "preload": "latest",
        "video_data_path": "face_emo_rec/data", 
        "training_data_folder": "face_emo_rec/data/training",
        "validation_data_folder": "face_emo_rec/data/validation",
        "eps": 1e-9,
        "margin": 1.0,
        "framesPerClip": 16, 
        "clips": 6,
        "localSpatTemprepresentation": 512,
        "clipLevelRepresentation": 8,
        "BiLSTM_hidden_size": 64,
        "BiLSTM_num_layers": 2,
        "out_classes": 8,
    }
    
def get_weights(config, epoch: str):
    model_folder = config['model_folder']
    os.makedirs(model_folder, exist_ok=True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)


def latest_weights(config):
    model_folder = config["model_folder"]
    os.makedirs(model_folder, exist_ok=True)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

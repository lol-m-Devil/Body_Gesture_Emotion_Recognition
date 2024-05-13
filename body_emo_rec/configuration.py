from pathlib import Path
import os

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 5,
        "lr": 0.001,
        "step_size": 50,  
        "gamma": 0.5,
        "train_test_split": 0.9,
        "model_folder": "body_emo_rec/weights",
        "model_basename": "EmoDet_",
        "experiment_name": "body_emo_rec/experiment_1",
        "preload": "latest",
        "video_data_path": "body_emo_rec/data", 
        "training_data_folder": "body_emo_rec/data/training",
        "validation_data_folder": "body_emo_rec/data/validation",
        "eps": 1e-9,
        "frames": 100,
        "margin": 1.0,
        "Joints": 33,
        "block1_filters": 128,
        "block2_filters": 256,
        "reduction_ratio": 16,
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

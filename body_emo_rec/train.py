import os
import shutil
import random
import torch
import configuration
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import torchmetrics
from tqdm import tqdm
import accm
from torch.optim.lr_scheduler import StepLR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def validation(model, validation_ds, device, global_step, writer, config):
    model.eval()

    expected = []
    predicted = []
    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc = f"Validation Epoch: {global_step:02d}")
        for batch in batch_iterator:
            body_joints, label = batch[0] , batch[1]
            body_joints = body_joints.permute(0, 2, 1, 3)
            
            output = model(body_joints.to(device))
            label = label.to(device)
            predicted_label = torch.argmax(output, dim = -1) + 1
            expected.append(label)
            predicted.append(predicted_label)
    
    expected = torch.stack(expected).flatten()
    predicted = torch.stack(predicted).flatten()        
    if writer:
        metric = torchmetrics.classification.Accuracy(task = "multiclass", num_classes = config["out_classes"]).to(device)
        acc = metric(predicted, expected)
        writer.add_scalar('validation accuracy', acc, global_step)
        writer.flush()

def directory_contains_folder(directory_path):
    # List all files and directories in the specified directory
    files_and_directories = os.listdir(directory_path)
    
    # Check if any of the entries in the list are directories
    for entry in files_and_directories:
        if os.path.isdir(os.path.join(directory_path, entry)):
            # If a directory is found, return True
            return True
    
    # If no directories are found, return False
    return False 

def get_ds(config):
    input_path = config["video_data_path"]

    train_folder = config["training_data_folder"]
    val_folder = config["validation_data_folder"]
    
    train_ratio = config["train_test_split"]

    
    # Move the videos to the appropriate folders
    if not directory_contains_folder(input_path):
        # Get the list of video files in the folder
        video_files = os.listdir(input_path)
        random.shuffle(video_files)  # Shuffle the list randomly

        # Calculate the number of videos for training and validation
        num_train = int(len(video_files) * train_ratio)
        # Create the training and validation folders if they don't exist
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        for i, video_file in enumerate(video_files):
            src = os.path.join(input_path, video_file)
            if i < num_train:
                dst = os.path.join(train_folder, video_file)
            else:
                dst = os.path.join(val_folder, video_file)
            shutil.move(src, dst)
    
    training_ds = dataset.VideoDataset(train_folder, config)
    validation_ds = dataset.VideoDataset(val_folder, config)
                
    training_dataloader = DataLoader(training_ds, batch_size = config["batch_size"], shuffle = True)
    validation_dataloader = DataLoader(validation_ds, batch_size = 1, shuffle = True)
    
    return training_dataloader, validation_dataloader
            
def get_model(config):
    
    model = accm.Architecture(config)
    return model

def train_model(config):
    
     # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("Device name: CPU")    
    device = torch.device(device)   
    
    training_dl, validation_dl = get_ds(config)   
    
    model = get_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])

    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    writer = SummaryWriter(config['experiment_name'])

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    modelFilename = None
    if preload == 'latest':
        modelFilename = configuration.latest_weights(config)
    elif preload:
        modelFilename = configuration.get_weights(config, preload)
    
    if modelFilename:
        print(f'Preloaded the weights of the model {modelFilename}')
        state = torch.load(modelFilename)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dictionary'])
        model.load_state_dict(state['model_state_dictionary'])
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(training_dl, desc = f"Processing Epoch: {epoch:02d}")
        for batch in batch_iterator:
            body_joints, label = batch[0] , batch[1]
            body_joints = body_joints.permute(0, 2, 1, 3)
            body_joints = torch.tensor(body_joints)
            output = model(body_joints.to(device))
        
            label = label.to(device)
            print("Forward Completed")
            loss = loss_fn(output, label)    
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            print(f"Loss Calculated{loss.item():6.3f}")
            
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            global_step += 1
            loss.backward()
            print("Backprop Completed")
            optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            print("Updation Step Completed")
            

            scheduler.step()
            
        
        #save your model
        model_filename = configuration.get_weights(config, f"{epoch:02d}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dictionary': model.state_dict(),
            'optimizer_state_dictionary': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
        #run the validation and log the details using tensorboard!
        validation(model, validation_dl, device, epoch, writer, config)
        

    writer.close()
    
if __name__ == '__main__':
    config = configuration.get_config()
    train_model(config)        
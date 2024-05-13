#Emotion Recognition:
Worked on implementation of the paper titled: "Learning facial expression and body gesture visual information for video emotion recognition" by Jie Wei, Guanyu Hu, Xinyu Yang, Anh Tuan Luu, Yizhuo Dong

#File Formats:

1. Dataset
   - Store all the videos in a folder named "data". 
   - The code has assumed each video file is named using the following convention: 
    - The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 
    - Filename identifiers 
        Modality (01 = full-AV). # Only considered a full av file. 
        Vocal channel (01 = speech, 02 = song).
        Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        Repetition (01 = 1st repetition, 02 = 2nd repetition).
        Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
   - Either change the video file names for the dataset used according to the given convention or change the code according to the video file names in dataset.py
     
2. If any folder path changes or hyperparameter changes are required, refer to the configuration.py file in both face_emo_rec and body_emo_rec


# To run this code: Run the train.py file which will run the code and run all processes in order to train the model.

# Note, the paper involves rcognizing emotions via two ways body gestures(body_emo_rec) and facial expressions(face_emo_rec).

# Important Files in Folder - body_emo_rec:

1. accm.py
    - contains architecture of ACCM
    
2. body_models.py
    - Implementation of ConvolutionalBlock, Channelwise ConvolutionalBlock, SE Attention mechanism.
    
3. poses.py
    - Implementation of a function titled bodyJoints which given a videoPath, returns a tensor of body joints of size 33.
    
4. dataset.py
    - we create a custom class: VideoDataset, which takes path to the dataset and returns a tensor of body joints and the corresponding label.
    
5. configuration.py
    - contains file paths and hyperparameter values
    
6. train.py
    - sets the device to 'GPU', if available.
    - splits data into training and validation.
    - gets model using get_model function.
    - uses traditional CrossEntropyLoss.
    - saves model weights after each epoch.
    - Main file that trains the model. 

# Important Files in Folder - face_emo_rec:

1. sistcm_bilstm.py
    - contains architecture of SISTCM-Resnet and BILSTM combined together
    
2. face_models.py
    - Implementation of Sistcm block, SISTCM resnet, Feature fusion in Sistcm, Bilstm.
    
3. face_detector.py
    - Implementation of a function which given a videoPath, returns an updated clip of image in a square shape(224x224).
    - Uses common.py and model- DBFace to detect face in the video.
     
4. dataset.py
    - we create a custom class: VideoDataset, which takes path to the dataset and returns a tensor of body joints and the corresponding label.
    
5. configuration.py
    - contains file paths and hyperparameter values
    
6. train.py
    - sets the device to 'GPU', if available.
    - splits data into training and validation.
    - gets model using get_model function.
    - uses traditional CrossEntropyLoss.
    - saves model weights after each epoch.
    - Main file that trains the model.     

NOTE: To visualize the loss and validation accuracy, we have created a writer which writes in the config["experiment_1"] file path.
To do this, after the code has successfully run, run the following command in your Python terminal:
tensorboard --logdir="absolute_path_to_experiment_1_folder"

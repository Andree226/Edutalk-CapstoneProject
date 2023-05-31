import librosa
import librosa.display as dsp
from IPython.display import Audio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_style("dark")

def get_audio(digit=0):
    # Audio Sample Directory
    sample = np.random.randint(1,60)
    # Index of Audio
    index = np.random.randint(1,5)
    
    # Modified file location
    if sample<10:
        file = f"../input/audio-mnist/data/0{sample}/{digit}_0{sample}_{index}.wav"
    else:
        file = f"../input/audio-mnist/data/{sample}/{digit}_{sample}_{index}.wav"

    
    # Get Audio from the location
    data,sample_rate = librosa.load(file)
    
    # Plot the audio wave
    dsp.waveshow(data,sr=sample_rate)
    plt.show()
    
    # Show the widget
    return Audio(data=data,rate=sample_rate)

def get_audio_raw(digit=0):
    # Audio Sample Directory
    sample = np.random.randint(1,60)
    # Index of Audio
    index = np.random.randint(1,5)
    
    # Modified file location
    if sample<10:
        file = f"../input/audio-mnist/data/0{sample}/{digit}_0{sample}_{index}.wav"
    else:
        file = f"../input/audio-mnist/data/{sample}/{digit}_{sample}_{index}.wav"

    
    # Get Audio from the location
    data,sample_rate = librosa.load(file)

    # Return audio
    return data,sample_rate
 def spectogram_of(digit):
    # Read the audio file
    data,sr = get_audio_raw(digit)
    # Apply Short-Time-Fourier-Transformer to transform data
    D = librosa.stft(data)
    # Converting frequency to decible
    S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
    # Plot the transformed data
    librosa.display.specshow(S_db,x_axis='time',y_axis='log')
    plt.show()
   
 # Creating subplots
fig,ax = plt.subplots(5,2,figsize=(15,30))

# Initializing row and column variables for subplots
row = 0
column = 0

for digit in range(10):  
    # Read the audio file
    data,sr = get_audio_raw(digit)
    # Apply Short-Time-Fourier-Transformer to transform data
    D = librosa.stft(data)
    # Converting frequency to decible
    S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
    # Plot the transformed data
    ax[row,column].set_title(f"Spectogram of digit {digit}")
    librosa.display.specshow(S_db,x_axis='time',y_axis='log',ax=ax[row,column])
    
    # Conditions for positioning of the plots
    if column == 1:
        column = 0
        row += 1
    else:
        column+=1
        
    
plt.tight_layout(pad=3)   
plt.show()

# will take a audio file as input and output extracted features using MEL_FREQUENCY CEPSTRAL COEFFICIENT
def extract_features(file):
    # Load audio and sample rate of audio
    audio,sample_rate = librosa.load(file)
    # Extract features using mel-frequency coefficient
    extracted_features = librosa.feature.mfcc(y=audio,
                                              sr=sample_rate,
                                              n_mfcc=40)
    
    # Scale the extracted features
    extracted_features = np.mean(extracted_features.T,axis=0)
    # Return the extracted features
    return extracted_features


def preprocess_and_create_dataset():
    # Path of folder where the audio files are present
    root_folder_path = "../input/audio-mnist/data/"
    # Empth List to create dataset
    dataset = []
    
    # Iterating through folders where each folder has audio of each digit
    for folder in tqdm(range(1,61),colour='green'):
        if folder<10:
            # Path of the folder
            folder = os.path.join(root_folder_path,"0"+str(folder))
        else:
            folder = os.path.join(root_folder_path,str(folder))
            
        # Iterate through each file of the present folder
        for file in tqdm(os.listdir(folder),colour='blue'):
            # Path of the file
            abs_file_path = os.path.join(folder,file)
            # Pass path of file to extracted_features() function to create features
            extracted_features = extract_features(abs_file_path) 
            # Class of the audio,i.e., the digit it represents
            class_label = file[0]
            
            # Append a list where the feature represents a column and class of the digit represents another column
            dataset.append([extracted_features,class_label])
    
    # After iterating through all the folder convert the list to a dataframe
    print("Extracted Features and Created Dataset Successfully !!")
    return pd.DataFrame(dataset,columns=['features','class'])
    
 # Create the dataset by calling the function
dataset = preprocess_and_create_dataset()
# A function which return MFCC
def extract_features_without_scaling(audio_data,sample_rate):
    # Extract features using mel-frequency coefficient
    extracted_features = librosa.feature.mfcc(y=audio_data,
                                              sr=sample_rate,
                                              n_mfcc=40)
    
    # Return Without Scaling
    return extracted_features
    
 # Creating subplots
fig,ax = plt.subplots(5,2,figsize=(15,30))

# Initializing row and column variables for subplots
row = 0
column = 0

for digit in range(10):  
    # Get Audio of different class(0-9)
    audio_data,sample_rate = get_audio_raw(digit)
    
    # Extract Its MFCC
    mfcc = extract_features_without_scaling(audio_data,sample_rate)
    print(f"Shape of MFCC of audio digit {digit} ---> ",mfcc.shape)
    
    # Display the plots and its title
    ax[row,column].set_title(f"MFCC of audio class {digit} across time")
    librosa.display.specshow(mfcc,sr=22050,ax=ax[row,column])
    
    # Set X-labels and y-labels
    ax[row,column].set_xlabel("Time")
    ax[row,column].set_ylabel("MFCC Coefficients")
    
    # Conditions for positioning of the plots
    if column == 1:
        column = 0
        row += 1
    else:
        column+=1
        
    
plt.tight_layout(pad=3)   
plt.show()

# Import Train Test Split
from sklearn.model_selection import train_test_split
# Seperate the audio and its class as X and Y
X = np.array(dataset['features'].to_list())
Y = np.array(dataset['class'].to_list())

# Create train set and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,shuffle=True,random_state=8)

# Import create an ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# To create a checkpoint and save the best model
from tensorflow.keras.callbacks import ModelCheckpoint

# To load the model
from tensorflow.keras.models import load_model

# To check the metrics of the model
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# Crete a Sequential Object
model = Sequential()
# Add first layer with 100 neurons to the sequental object
model.add(Dense(100,input_shape=(40,),activation='relu'))
# Add second layer with 200 neurons to the sequental object
model.add(Dense(100,activation='relu'))
# Add third later with 100 neurons to the sequental object
model.add(Dense(100,activation='relu'))

# Output layer With 10 neurons as it has 10 classes
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')
              
# Set the number of epochs for training
num_epochs = 100
# Set the batch size for training
batch_size = 32

# Fit the model
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=num_epochs,batch_size=batch_size,verbose=1)

# Make predictions on test set
Y_pred = model.predict(X_test)
Y_pred = [np.argmax(i) for i in Y_pred]

# Print the metrics
print(classification_report(Y_test,Y_pred))

# Set style as dark
sns.set_style("dark")
# Set figure size
plt.figure(figsize=(15,8))

# Plot the title
plt.title("CONFUSION MATRIX FOR MNIST AUDIO PREDICTION")
# Confusion matrix
cm = confusion_matrix([int(x) for x in Y_test],Y_pred)
# Plot confusion matrix as heatmap
sns.heatmap(cm, annot=True, cmap="cool", fmt='g', cbar=False)
# Set x-label and y-label
plt.xlabel("ACTUAL VALUES")
plt.ylabel("PREDICTED VALUES")

# Plot the plot
plt.show()

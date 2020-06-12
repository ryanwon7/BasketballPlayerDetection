# Tracking and Classifying NBA Players in Basketball Videos
#### Ryan Won CS83
##Important to Note
The subdirectory cnn_models is not included in this repo as it contains the CNN outputs, which are too large to be stored 
on github. The zip file containing them and the information/data about them are attached in the BBLearn submission.
Additionally, the project requires the file "mask_rcnn_coco.h5" to run training model. This file is avaialble in the 
RCNN repository here: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

To look at the resulting video from this project, look under /videos/output.avi
## Directory and File Description
The directory contains a multitude of scripts that are placed in three sub-directories.
* **/cnn_frames** - This subdirectory contains all the still images(frames) that have the prediction bounding boxes applied
to them.
* **/cnn_models** - In Github, this directory not present, as this subdirectory houses the five output models 
from the training model, which are each very large in memory. Also contains an info.txt file with information and data
on each epoch.
* **/raw_frames** - Contains the still images(frames) that are directly taken from the input test video.
* **/training_set** - Contains the still images(frames) and annotations for the training images.
  * **/annot** - Contains the XML files of annotations of each training image.
  * **/images** - Contains the still images(frames) that are directly taken from the input training video.
* **/videos** - Contains the training input video, test input video, and resulting video (output.avi).
* **model_config.py** - Contains the Database class and configuration elements necessary to create training data from 
images and XML files.
* **player_detection.py** - The main script that runs the player detection test. Takes arguments of input test video file
and the framerate to use.
* **model_config.py** - Script used to train the model. Takes argument of epochs.

## Library Requirements
Conda was used to create a python environment running on Python 3.6. Below is the list of dependencies for this project.
# platform: win-64
* bleach=1.5.0=pypi_0
* certifi=2020.4.5.1=py36_0
* cycler=0.10.0=pypi_0
* decorator=4.4.2=pypi_0
* enum34=1.1.10=pypi_0
* h5py=2.10.0=pypi_0
* html5lib=0.9999999=pypi_0
* imageio=2.8.0=pypi_0
* importlib-metadata=1.6.1=pypi_0
* keras=2.1.5=pypi_0
* keras-applications=1.0.8=pypi_0
* keras-preprocessing=1.1.2=pypi_0
* kiwisolver=1.2.0=pypi_0
* markdown=3.2.2=pypi_0
* matplotlib=3.2.1=pypi_0
* networkx=2.4=pypi_0
* numpy=1.18.5=pypi_0
* opencv-python=4.2.0.34=pypi_0
* pillow=7.1.2=pypi_0
* pip=20.0.2=py36_3
* protobuf=3.12.2=pypi_0
* pyparsing=2.4.7=pypi_0
* python=3.6.10=h9f7ef89_2
* python-dateutil=2.8.1=pypi_0
* pywavelets=1.1.1=pypi_0
* pyyaml=5.3.1=pypi_0
* scikit-image=0.17.2=pypi_0
* scipy=1.4.1=pypi_0 
* setuptools=47.1.1=py36_0
* six=1.15.0=pypi_0
* sqlite=3.31.1=h2a8f88b_1
* tensorflow-gpu=1.4.0=pypi_0
* tensorflow-tensorboard=0.4.0=pypi_0
* tifffile=2020.6.3=pypi_0
* tqdm=4.46.1=pypi_0
* vc=14.1=h0510ff6_4
* vs2015_runtime=14.16.27012=hf0eaf9b_2
* werkzeug=1.0.1=pypi_0
* wheel=0.34.2=py36_0
* wincertstore=0.2=py36h7fe50ca_0
* zipp=3.1.0=pypi_0
* zlib=1.2.11=h62dcd97_4

## How to Run
First, run the **train_cnn.py** script to train your images. This requires you to first run and split the images from 
whichever training video you want to use. Then, once your neural networks have been trained, select which output node you
want to use and set that in the parameters in player_detection.py. Use your test video with player_detection.py to then
get the desired output video.
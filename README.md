# DL-instagram-classification

PRETRAINED MODELS<br />
To run the pretrained CNN models the preTrainedDeployment.ipynb notebook will walk you through all steps. This notebook will download the images and creat directories for the images. It will also download the .h5 models and will continue to walk you through testing them on the images. Finally at the end of the notebook you can test individual images and upload custom images.

TRAINING MODELS FROM SCRATCH<br />
To train these models from scratch you can run the fullRun.ipynb notebook. This notebook will also download the images and create directories. It will then walk you through running the individual model.py files. At the end there are loops that will output into text files that can be used in the crossValidationVisual.ipynb notebook.

MODEL TRAINING VISUALIZATION<br />
After collecting the necessary text files from fullRun.ipynb, run the crossValidationVisual.ipynb notebook which will load in the text files and display the validation accuraccies across epochs with confidence intervals.

Code built on top of previous work by https://www.kaggle.com/code/gpiosenka/f1-score-98

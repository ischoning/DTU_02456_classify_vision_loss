# DTU_02456_classify_vision_loss
An LSTM with a linear layer that classifies between central and peripheral vision loss with accuracy of approximately 70% given positional and velocity features of gaze data recorded on Pupil Invisible glasses (66Hz).

**Constants.py**: Holds the global parameters used in the model and pre-processing of the data.

**data_utils.py**: A customized version inspired by Pytorch's datautils that loads data to be compatible with batches.

**main.ipynb**: A jupyter notebook of the full code, including pre-processing of the data, the model, and training and testing the model.

This repo was created solely as part of the final project report for 02456 Deep Learning at the Technical University of Denmark (fall semester 2021). That is why this repo is made public, so the professors can view it. However, the data is IP and therefore cannot be accessed when run outside of a private server. Therefore, this repo is for viewing purposes only. If you are a professor or grader for this course and need to run the model, please contact me s202576@student.dtu.dk.

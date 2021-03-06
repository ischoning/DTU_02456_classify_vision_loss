# DTU_02456_classify_vision_loss
A LSTM with linear input and output layers that classifies between central and peripheral vision loss with accuracy of approximately 70% given positional and velocity features of gaze data recorded on Pupil Invisible glasses (66Hz).

**Constants.py**: Holds the global parameters used in the model and pre-processing of the data.

**data_utils.py**: A customized version inspired by Pytorch's datautils that loads data to be compatible with batches.

**main.ipynb**: A jupyter notebook of the full code, including pre-processing of the data, the model, and training and testing the model.

**02456-poster-final.pdf**: A poster of preliminary results published before the report and final code.

**Classify_using_distribution_params**: Contains code for a CNN that trains given inputs being the means and variances of feature distributions from each sequence rather than the sequences themselves. This was just a test out of curiosity.

This repo was created solely as part of the final project report for 02456 Deep Learning at the Technical University of Denmark (fall semester 2021). That is why this repo is made public, so the professors can view it. However, the data is IP and therefore cannot be accessed when run outside of a private server. Therefore, this repo is for viewing purposes only. If you are a professor or grader for this course and need to run the model, please contact me s202576@student.dtu.dk.

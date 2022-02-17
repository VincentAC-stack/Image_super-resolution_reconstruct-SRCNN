# Image_super-resolution_reconstruct-SRCNN
This is the **[paper](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)** of Image Super-Resolution Convolutional Neural Network (SRCNN).

# Basic Function of each file
- step1-pre is the preprocessing program: the output is h5 format files of train and test.
- step2-train is the program for training the model: the output is the trained model parameters saved in the checkpoint folder.
- step3-test is the prediction program: the input is an h5 file of a picture, and the output pnsr value. And save the input image, label image, and prediction image in the sample for viewing.
- checkpoint folder: saved model parameters.
- h5 folder: h5 files of the stored train and test training sets.
- sample folder: saves the input, label, and prediction images.

# Summary
Here the model parameters have been trained and saved, run strp-test directly to load the parameters saved in the checkpoint folder, and finally get the pnsr value. The input is a picture in the set folder in the test folder (there are originally five pictures, take one for the experiment first), here is the picture of woman. The obtained pnsr is 29.09db, which is better than the 28.56db of bicubic interpolation, but still far from the ideal effect of 30.92db. By observing the pictures in the sample folder with the naked eye, it is obvious that the predicted pictures are darker than the input and label. I don't know if the reason is here. The report of SRCNN is as follows: **[SRCNN](https://github.com/VincentAC-stack/Image_super-resolution_reconstruct-SRCNN/blob/main/SRCNN%20report.pdf)**.

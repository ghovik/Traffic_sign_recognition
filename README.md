# Traffic Sign Classification

Pipeline:

* Load the [GTSRB dataset](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)
* Explore, summarize and visualize the dataset
* Pre-process the dataset images for better training result
  * Methods involved: grayscale conversion, contrast limited adaptive histogram equalization ([CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)), zero-mean normalization, down and/or up-sampling, shuffling. 
* Design, train and test different model architectures
* Use trained model to make predictions on new traffic sign images



### Introduction

Recognition of traffic signs is a challenging task in many real-world applications, such as autonomous driving, which is one of the hottest topics right now in the world. A typical traffic sign recognition system consists of two main parts, traffic sign detection and recognition. This project is focused on the recognition. The idea of this project is inspired by a multi-class competition, called German Traffic Sign Recognition Benchmark (GTSRB), proposed at IJCNN in 2011. A dataset contains more then 50,000 traffic sign images is provided by the competition. The class frequencies are unbalanced due to the fact that the traffic signs have a wide range of variations in terms of color, shape, and the presence of figures and text. Consequently, several pre-process methods based on computer vision have been used to provide images with better quality. After that, two CNN classifiers based on the famous LeNet are implemented. After training, both classifiers provide more than 92% validation accuracy.



### Dataset summary

- The size of training set is 37799
- The size of the validation set is 4410
- The size of test set is 12630
- The shape of a traffic sign image is (32, 32, 3)
- The number of unique classes/labels in the data set is 43
- Image sizes vary from 15x15 to 250x250 pixels.

The histogram below shows the distribution of each traffic sign class.

![](images\hist.png)

Here shows one example per traffic sign type.

![](images\sign_examples.png)



### Pre-processing

1. Convert to grayscale, as pointed out in [LeCun's paper](https://www.researchgate.net/profile/Yann_Lecun/publication/224260345_Traffic_sign_recognition_with_multi-scale_Convolutional_Networks/links/0912f50f9e763201ab000000/Traffic-sign-recognition-with-multi-scale-Convolutional-Networks.pdf) that the CNN classifier is not sensitive to colored images.
2. Apply CLAHE to combat poor quality of the images. By doing so, the contrast in the images were improved, but the noise in relatively homogeneous regions is not over-amplified.
3. Normalization. The pixel values of the images (0 to 256) are normalized to zero-mean (-1 to 1).
4. Up-sample or down-sample the image sizes to 32x32.
5. Shuffle the images, since the ordering of the images has a huge impact of how well the model trains.

Below is an example of step by step pre-processed images. From left to right: original, grayscaled, CLAHE applied, and normalized.



![](images\pre-process.png)



### CNN Architecture Design

Two CNNs were used to train, the first is the famous LeNet, and the second is a modified version of it. Detailed implementations can be found in the Jupyter notebook file.

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |              32x32x3 RGB image              |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6  |
|      RELU       |                                             |
|   Max pooling   |        2x2 stride,  outputs 14x14x6         |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                             |
|   Max pooling   |         2x2 stride, outputs 5x5x16          |
|     Flatten     |                 Outputs 400                 |
| Fully connected |                 Output 120                  |
|      RELU       |                                             |
|     Dropout     |              Dropout rate = 1               |
| Fully connected |                  Output 84                  |
|      RELU       |                                             |
|     Dropout     |              Dropout rate = 1               |
| Fully connected |                  Output 43                  |



### Results

Batch size: 128

epochs: 100

Learning rate: 0.0015

Drop rate: 0.15

$\mu$: 0

$\sigma$: 0.1

| Model          | Methods                         | Validation Accuracy |
| -------------- | ------------------------------- | ------------------- |
| LeNet          | Normalization                   | 0.884               |
| LeNet          | Grayscale, Normalization        | 0.907               |
| LeNet          | Grayscale, CLAHE, Normalization | 0.924               |
| Modified LeNet | Normalization                   | 0.943               |
| Modified LeNet | Grayscale, Normalization        | 0.950               |
| Modified LeNet | Grayscale, CLAHE, Normalization | 0.968               |



### Test on New Images

A few traffic sign images from the web were used for testing our model. These images have certain problems that increase the difficulty to classify, they might be too dark, too similar to some other signs, or the view is partially obstructed. 

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.918. However this result can be treated just as a reference since the size of the samples is too small.
[![OpenCV](https://img.shields.io/badge/dependency-OpenCV-green.svg)](https://github.com/opencv/opencv)
[![Licence](https://img.shields.io/badge/license-BSD-yellow.svg)](http://opencv.org/license.html)
[![Language](https://img.shields.io/badge/language-C%2B%2B-brightgreen.svg)](https://en.wikipedia.org/wiki/C%2B%2B)
# GSoC 2017: Face Alignment Module based on OpenCV
This project is an implementation of state of the art Face alignment Technique proposed by Vahid Kazemi.

## Project Description
The GSoC proposal can be found on [Drive](https://drive.google.com/open?id=10LIfoV-pRIjoH-xNmLwaQOANavfMYT-5cS-4FpwTSPE).
We aim to implement the paper [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf) using OpenCV.

<a href="https://www.youtube.com/watch?v=IOlAuQkZ4fM" target="_blank"><img src="https://github.com/sakshamgupta006/face_align/blob/master/Results/result_3.png" 
alt="FACE_ALIGNMENT" width="360" height="360" border="10" /></a>
<a href="https://www.youtube.com/watch?v=IOlAuQkZ4fM" target="_blank"><img src="https://github.com/sakshamgupta006/face_align/blob/master/Results/result_4.png" 
alt="FACE_ALIGNMENT" width="360" height="360" border="10" /></a>
<a href="https://www.youtube.com/watch?v=IOlAuQkZ4fM" target="_blank"><img src="https://github.com/sakshamgupta006/face_align/blob/master/Results/Filter_Result.png" 
alt="FACE_ALIGNMENT" width="360" height="360" border="10" /></a>

## Pre-requisites
The module has been tested on **Ubuntu 16.04** and **Ubuntu 14.04** with **OpenCV 3.2** , but it should be easy to compile it on other platforms.

### C++11 or C++0x Compiler
The code is based on C++, so a C++ compiler is must.

### OpenCV 3.x.x
The code is compatible with every version of OpenCV after version 3.0.0 including 3.0.0 .

### Face Detector Trained Model
The default trained tree was trained on OpenCV's **HAAR cascade** which can be downloaded from [OpenCV's repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml). The model is also provided in the **data** folder.

## Building Face-Alignment Module and Examples

Clone the repository:
```
https://github.com/sakshamgupta006/face_align.git
```
Then follow the following steps:
```
cd face_align
mkdir build
cd build
cmake ..
make
```

## Running Examples
The module comes with examples demonstrating the results on **Video, Image, Multiple Images, External Capturing Devices.** Additionally, the module also includes examples to:
* **Train** Face Alignment on new dataset and parameters.
* **Test Accuracy** of the trained model.
* **Landmarks Detection** from the trained model
* **A Snapchat Filter** Like application 

### Training Example
![](https://media.giphy.com/media/xUNemUdwFSCRCWTq4U/giphy.gif)
```
./Train_ex -cascade=<location of HAAR cascade model> -path=<path to the dataset> <Output Name of Trained Model>
```
To modify the parameters for Training run the following command
```
./Train_ex -cascade=<location of HAAR cascade model> -path=<path to the dataset> -landmarks=<number of Landmarks> -oversampling=<Over sampling Amount> -learningrate=<Learning Rate> -cascadedepth=<> -treedepth=<> -treespercascade=<> -testcoordinates=<> -numtestsplits=<> -lambda=<> -samples<Number of images from dataset> <Output Name of Trained Model>

```
A detailed description of each parameter is given in subsequent sections
or you can simply run
```
./Train_ex -path=<path to the dataset>
```
This will take the default values and an output file with name **68_landmarks_face_align.dat** will be generated after training.

### Testing Accuracy Example
```
./Test_accuracy_ex -cascade=../data/haarcascade_frontalface_alt2.xml -path=../data/dataset/ -model=../data/68_landmarks_face_align.dat -samples=500
```

### Face Landmark Detection Example
#### For Videos
```
./Landmark_detection_ex -cascade=<location of HAAR cascade model> -model=<path to trained model> <path to video file / external capturing device id>
```
If no <path to video file> is defined the code will capture the web camera, if available.

#### For Images
```
./Landmark_detection_ex -cascade=<location of HAAR cascade model> -model=<path to trained model> <path to image / path to txt file containig images location>
```
or you can simply run
```
./Landmark_detection_ex <path to image / path to txt file containig images location>
```
This will take the default values of cascade and trained model.
For better results you can download and use the heavier model from [Here](https://drive.google.com/open?id=0B3qCZz_EciMmcU9OcEhiWmZSQms)

### Filter Application Example
```
./Filter_ex -filter=<enum value of corresponding filter> <Path to image/video input>
```
## Dataset
The module was trained on a combination of HELEN, LFPW, iBug, and AFW datasets and their mirror images. The combined dataset can be downloaded from [Drive](https://drive.google.com/file/d/0B8t1D28N36RXTGIxd1o5QnBRT1E/view?usp=sharing).

## Mentor
I am being mentored by [Simon Niklaus](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjLtZSM-bPUAhXIgI8KHXKwCuoQFggqMAE&url=http%3A%2F%2Fwww.sniklaus.com%2F&usg=AFQjCNFbkMT3hq7F2D-SSMwDmVMIkBXZAg&sig2=wHJv3K7Zb5IvDuFeGWY7yg) from Portland State University.

## Documentation
Detailed explaination of each function can be found inside the **include/train_shape.hpp** file alongside the concerend function.

## Timeline
A detailed timeline of the project can be found on [Google Calander](https://calendar.google.com/calendar/embed?src=q5686lb1opb5kfrqatjslo2060%40group.calendar.google.com&ctz=Asia/Calcutta)

## Blog
Blog regarding the GScC project journey can be found [Here](saksham-gsoc2017.blogspot.com).

## Description of Training Parameters
* **cascade:** Face Detector Model file path. For best results use Haar cascade frontalface alt2 xml model
* **path:** Dataset folder path
* **landmarks:** Number of Landmarks to be trained on.
* **oversampling:** Number of different initializations for each training image.
* **learningrate:** This factor determines the rate of change of learned values. Used to modify the leaves of regression trees.
* **cascadedepth:** Number of Cascade's for the Gradient boosting function
* **treedepth:** Depth of each tree in the cascade
* **treespercascade:** Number of trees in each cascade 
* **testcoordinates** Number of test coordinates where splits will be tested
* **numtestsplits** Number of good splits to consider.
* **lambda** The priori variable to decide a good random split feature
* **samples** Number of samples/images to train on.

## Training on a custom Dataset
To train the module on custom dataset or specific number of Landmarks use the sample_Face_Train.cpp and modify the dataset reading functions. The trainCascade function expects an **unordered_map** of < string, vector<Point2f> > here the string is the key of the map and it stores the location of the image correspoinding to which the vector Point2f contains the Landmarks. So, to train on any custom dataset just provide the **unordered_map** with the above mentioned properties.

## Future Work

* Speed-Up the module
* Provide support for Deep Learned Models like tweaked CNN's for facial Landmark Detection
* Improve the accuracy using the immediate shape instead of the mean shape every time
* More applications like: Face swapping, Face time lapse

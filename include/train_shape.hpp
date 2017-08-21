/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008-2013, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Itseez Inc. may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef __OPENCV_TRAIN_SHAPE_HPP__
#define __OPENCV_TRAIN_SHAPE_HPP__

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <bits/stdc++.h>

using namespace std;

namespace cv{
//! struct for creating a split Feature in the regression Tree
struct splitFeature
{
    //! index which stores the left subtree value of the split for the regression tree
    unsigned long idx1;
    //! index which stores the right subtree value of the split for the regression tree
    unsigned long idx2;
    //! threshold which decides wheather the split is good or not
    float thresh;
};
//! struct for holding a Regression Tree
struct regressionTree
{
    //! vector that contains split features which forms the decision nodes of the tree
    vector<splitFeature> split;
    //! vector that contains the annotation values provided by the Regression Tree at the terminal nodes
    vector< vector<Point2f> > leaves;
};
//! struct for holding the training samples attributes during training of Regression Tree's
struct trainSample
{
    //! Mat object to store the image from the dataset
    Mat img;
    //! vector to store faces detected using any standard facial detector
    vector<Rect> rect;
    //! vector to store final annotations of the face
    vector<Point2f> targetShape;
    //! vector that will contain the current annotations when regression tree is being trained
    vector<Point2f> currentShape;
    //! vector that will contain the residual annotations that is obtained using current and target annotations
    vector<Point2f> residualShape;
    //! vector to store the pixel values at the desired annotations locations
    vector<double> pixelValues;
    //! vector to strore test pixel coordinates at each cascade level
    vector<Point2f> pixelCoordinates;
};


class KazemiFaceAlignImpl
{
    public:

        // TRAINING FACE ALIGNMENT FUNCTIONS

        /** @brief Set's the number of Landmarks to detect in the face.  
        * @param numberLandmarks A variable of type std::unsigned long int contains the number of landmarks to be detected
        */
        void setnumLandmarks(unsigned long numberLandmarks);
        
        /** @returns A std::unsigned long int containing the number of Landmarks to be detected.*/
        unsigned long getnumLandmarks() const {return numLandmarks;}
        
        /** @brief Set's the Over sampling amount of each training image. 
        * @param oversampling A std::unsigned long int containing the the number of initializations of each training image
        */
        void setOverSampling(unsigned long oversampling);

        /** @return A std::unsigned long int containing the over sampling amount used.*/
        unsigned long getOverSampling() const { return oversamplingAmount;}

        /** @brief Set's the Learning Rate used during the training 
        * @param A variable of type std::float containing new learning Rate
        */
        void setLearningRate(float learingrate);

        /** @returns A std::float denoting the learning rate used during training*/
        float getLearningRate() const{ return learningRate;}
        
        /** @brief Set's the Cascde Depth or number of Cascade in the forest 
        * @param A variable of type std::unsigned long int containing the new cascade depth 
        */
        void setCascadeDepth(unsigned long cascadedepth);
        
        /** @returns A std::unsigned long int containing the cascade depth used during training*/
        unsigned long getCascadeDepth() const {return cascadeDepth;}
        
        /** @brief Set's the Depth of each Tree in the cascade
        * @param A variable of type std::unsigned long int containng the new tree depth
        */
        void setTreeDepth(unsigned long treedepth);
        
        /** @returns A std::unsigned long int containing the tree depth value used during training*/
        unsigned long getTreeDepth() const {return treeDepth;}

        /** @brief Set's the number of trees per cascade
        * @param treespercascade A variable of type std::unsigned long int containing the new value of number of trees to be set per cascade
        */
        void setTreesPerCascade(unsigned long treespercascade);

        /** @returns A std::unsigned long int containing the number of trees per cascade*/
        unsigned long getTreesPerCascade() const { return numTreesperCascade;}

        /** @brief Set's number of Test Co-ordinates onto which the training will run
        * @param A variable of type std::unsinged long int containing the new number of test co-ordinates
        */
        void setTestCoordinates(unsigned long testcoordinates);

        /** @returns  A std::unsigned long int denoting the number of test co-ordinates used during training*/
        unsigned long getTestCoordinates() const { return numTestCoordinates;}

        /** @brief Set's the number of test splits that will be performed during training
        * @param A varaible of type std::unsigned long int containing the new number of test splits
        */
        void setTestSplits(unsigned long testsplits);

        /** @returns A std::unsigned long int containing the number of test splits performed during training*/
        unsigned long getTestSplits() const { return numTestSplits;}

        /** @brief Set's the value of Lambda to in the priori given by Vaheid Kazemi
        * @param A variable of type std::float containing the new value of Lambda
        */
        void setLambda(float Lambda);

        /** @returns A std::float denoting the value of lambda used during the training*/
        float getLambda() const { return lambda;}

        /** @brief Set's the number of images to be taken from the whole dataset
        * @param A std::unsigned long int denoting the number of images to be considered during training from the dataset
        */
        void setnumSamples(unsigned long numsamples);

        /** @returns A std::unsigned long int denoting the number of images considered during training from the dataset*/
        unsigned long getnumSamples() const {return numSamples;}

        /** @brief Given an index in an array this function returns the location of its left child
        * @param idx A std::unsigned long int denoting the index of node whose left child has to be determined
        * @returns A std::unsigned long int containing the location of left child 
        */
        unsigned long leftChild (unsigned long idx);
        
        /** @brief Given an index in an array this function returns the location of its right child
        * @param idx A std::unsigned long int denoting the index of node whose right child has to be determined
        * @returns A std::unsigned long int containing the location of right child 
        */
        unsigned long rightChild (unsigned long idx);
        
        
        /** @brief This function calculates euclid distance between two pixel co-ordinates
        * @param first A variable of type cv::Point2f containing the first co-ordinate
        * @param second A variable of type cv::Point2f containing the second co-ordinate
        * @returns A double variable containing euclid distance between the points
        */
        double getDistance(Point2f first , Point2f second);

        /** @brief Calculates pointwise Difference between two cv::Point2f vectors
        * @param input1 A vector of type cv:Point2f from which the other vector will be subtracted
        * @param input2 A vector of type cv::Point2f which will be subtracted from initial vector
        * @param output A vector of type cv::Point2f which stores the difference between the input vectors
        * @returns A boolean value: Returns true when the operation is successful else false
        */
        bool calcDiff(vector<Point2f>& input1, vector<Point2f>& input2, vector<Point2f>& output);
        
        /** @brief Calculates pointwise Sum between two cv::Point2f vectors
        * @param input1 A vector of type cv:Point2f to be added with the other vector
        * @param input2 A vector of type cv::Point2f which will be added with initial vector
        * @param output A vector of type cv::Point2f which stores the sum of the input vectors
        * @returns A boolean value: Returns true when the operation is successful else false
        */
        bool calcSum(vector<Point2f>& input1, vector<Point2f>& input2, vector<Point2f>& output);

        /** @brief A simplified standard function to call the detectMultiScale function for face detection  
        * @param image A variable of type cv::Mat denoting the image
        * @param cascade A CascadeClassifier already loaded in memory. Supports HAAR, LBP and HOG based classifiers
        * @returns A vector of type cv::Rect denoting the position of all the faces identified in the image given
        */
        vector<Rect> faceDetector(Mat image,CascadeClassifier& cascade);

        /** @brief Calculates the extreme bounds of Mean Shape to be used during the calculation of Affine Matrix 
        * @returns A boolean value: It return true when the mean shape's bound were calculated successfully and meanShapeRefernce 
        *variable is filled else it return false
        */
        bool calcMeanShapeBounds();
        
        /** @brief reads the file list(xml) created by imagelist_creator.cpp
        * @param names A vector of type cv::String into which names of respective files will be loaded 
        * @param annotation_path_prefix A variable of type std::string containing the path leading to the data which is to be augmented before the name
        * @returns A boolean Value: True when the reading operation is successful else false 
        */
        bool readAnnotationList(vector<cv::String>& names, string annotation_path_prefix);
        
        /** @brief Parse the txt file to extract image and its annotations
        * @param filepath A vector of type cv::String containing the path to the .txt/.pts files which have the location of Landmarks
        * @param landmarks An unordorderd map with key of type std::string denoting the image name and the data value of vector of cv::Point2f denoting the location of Landmarks 
        * @param path_prefix  A variable of type std::string containing the path leading to the data which is to be augmented before each filepath
        * @returns
        */
        bool readtxt(vector<cv::String>& filepath, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix);
        
        /** @brief This function reads the 68 Landmarks dataset
        * @param l A vector of type cv::String containing the path to the .txt/.pts files which have the location of Landmarks
        * @param landmarks An unordorderd map with key of type std::string denoting the image name and the data value of vector of cv::Point2f denoting the location of Landmarks 
        * @param path_prefix A variable of type std::string which stores any path to be augmented to access the .pts files
        * @returns A boolean value: True when the reading is completed successfully else false.
        */
        bool readnewdataset(vector<cv::String>& l, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix);

        /** @brief This function reads the mirrored 68 Landmarks dataset. The dataset can be downloaded from http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz.
        * @param l A vector of type cv::String containing the path to the .txt/.pts files which have the location of Landmarks
        * @param landmarks An unordorderd map with key of type std::string denoting the image name and the data value of vector of cv::Point2f denoting the location of Landmarks 
        * @param path_prefix A variable of type std::string which stores any path to be augmented to access the .pts files
        * @returns A boolean value: True when the reading is completed successfully else false.
        */
        bool readmirror(vector<cv::String>& l, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix);

        /** @brief This function is responsible of filling the data into samples, It calculates the Mean Shape and create a training Sample 
        *for each of the images, applying Face Detector and filling the target and current shape for each of the samples
        * @param samples A vector of training samples
        * @param landmarks An unordorderd map with key of type std::string denoting the image name and the data value of vector of cv::Point2f denoting the location of Landmarks 
        * @param cascade  A CascadeClassifier already loaded in memory. Supports HAAR, LBP and HOG based classifiers
        * @returns A boolean value: True denotes successful initialization of samples and False denotes some problem with data loading
        */
        bool fillData(vector<trainSample>& samples,std::unordered_map<string, vector<Point2f>>& landmarks,
                        CascadeClassifier& cascade);
        
        /** @brief This function Scale the data so that the processing can be fast and prediction is taken on the most optimal image size
        * @param trainimages A variable of type cv::Mat containing the image to be resized
        * @param trainlandmarks A vector of type cv::Point2f containing the target shape of the image
        * @param s A variable of type cv::Size representing the new size of the image
        * @returns A boolean value: True if the scaling is successful else false
        */
        bool scaleData(Mat& trainimages, vector<Point2f>& trainlandmarks, Size s);
        


        // REGRESSION TREE FUNCTIONS

        /** @brief Randomly Generates splits given a set of pixel co-ordinates use the priori given by the paper  
        * @param pixelCoordinates A vector of cv::Point2f containing the randomly generated pixel co-ordinates
        * @returns A splitFeature containing a randomly generated split using the priori
        */
        splitFeature randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates);
        
        /** @brief This function is responsible for choosing the best split feature from the randomly generated splits 
        * @param samples A vector of training samples
        * @param pixelCoordinates A vector of cv::Point2f containing the randomly generated pixel co-ordinates
        * @param start A variable of type std::unsigned long int denoting the starting index in the vector of training samples
        * @param end A variable of type std::unsigned long int denoting the ending index in the vector of training samples
        * @param sum A vector of type cv::Point2f denoting the sum of the Residual shapes of the samples
        * @param leftSum A vector of type cv::Point2f which is the LeftChild i.e. 2*i of the sum's index
        * @param rightSum A vector of type cv::Point2f which is the RightChild i.e. 2*i+1 of the sum's index
        * @returns A split Feature which is appended in the regression tree
        */
        splitFeature splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long start ,
                                    unsigned long end, vector<Point2f>& sum, vector<Point2f>& leftSum, vector<Point2f>& rightSum);
        
        /** @brief Samples the partition given a spit
        * @param split A variable of type splitFeature describing the split
        * @param samples A vector of training samples
        * @param start A variable of type unsigned long int depicting the starting index in training samples vector from where the split will initiate
        * @param end A variable of type unsigned long int depicting the ending index in training samples vector from where the split will end
        * @returns A variable of type unsigned long int depecting the middle index after the split
        */
        unsigned long partitionSamples(splitFeature split, vector<trainSample>& samples,
                                        unsigned long start, unsigned long end);
        /** @brief Find the nearest landmark's index in the mean shape from the given point
        * @param A variable of cv::Point2f containg the location of the point whose nearest landmark has to be calculated
        * @returns An std::unsigned long int denoting the index of the nearest landmark in mean shape
        */
        unsigned long findNearestLandmark(Point2f& pixelValue);
        
        /** @brief Calculates the Relative pixel in the sample image corresponding to the test pixel co-ordinates
        * @param sample A train sample with image, current and target shape filled
        * @returns A boolean value: True denotes successful calculation of relative pixel else false
        */
        bool calcRelativePixels(trainSample& sample);
        
        /** @brief Generate Test co-ordinates using Mean Shape bounds
        * @param pixelCoordinates A vector of vector of type cv::Point2f in which the generated test-coordinates will be filled
        * @returns A boolean Value: True if the generation of test cases were successful else false
        */
        bool generateTestCoordinates(vector< vector<Point2f> >& pixelCoordinates);
        
        /** @brief Normalize the Landmarks to a Rectangle of 1X1 unit
        * @param r A variable of type cv::Rect containing one of the face detector results
        * @returns A cv::Mat denoting the affine matrix from rectangle extremes to 1x1 unit rectangle 
        */
        Mat normalizing_tform(Rect& r);

        /** @brief Unormalize the Landmarks to the original domain of the image
        * @param r A variable of type cv::Rect containing one of the face detector results
        * @returns A cv::Mat denoting the affine matrix from 1x1 unit rectangle to rectangle extremes
        */
        Mat unnormalizing_tform(Rect& r);

        /** @brief This function extracts the pixel values corresponding to the test pixel coordinates generated earlier
        * @param sample A train sample with image, current and target shape filled
        * @returns A boolean Value : True denotes successful filling of the pixel values in the sample else false
        */
        bool extractPixelValues(trainSample &sample);

        /** @brief This function is responsible of building a single regression tree
        * @param samples A vector of type trainSample filled during fillData function
        * @param pixelCoordinates A vector of type cv::Point2f filled during the generateTestCoordinates function
        * @returns A regression Tree with splits and leaf values
        */
        regressionTree buildRegressionTree(vector<trainSample>& samples, vector<Point2f>& pixelCoordinates);

        /** @brief This function find the residual Shape, Mean Residual shape and is resposible for the GRADIENT BOOST ANALOGY
        * @param samples A vector of type trainSample filled during fillData function
        * @param pixelCoordinates A vector of type cv::Point2f filled during the generateTestCoordinates function
        * @returns A vector of regression tree(forest)
        */
        vector<regressionTree> gradientBoosting(vector<trainSample>& samples, vector<Point2f>& pixelCoordinates);

        /** @brief Intitiates the training of Cascade 
        * @param landmarks An unordorderd map with key of type std::string denoting the image name and the data value of vector of cv::Point2f denoting the location of Landmarks 
        * @param cascade  A CascadeClassifier already loaded in memory. Supports HAAR, LBP and HOG based classifiers
        * @param outputName A variable of type std::string denoting the name of the trained model
        * @returns A boolean value: True if the training is successful else false
        */
        bool trainCascade(std::unordered_map<string, vector<Point2f>>& landmarks, CascadeClassifier& cascade, string outputName);
        
        /** @brief A Parallelized Implementation of Calculating Relative Pixels
        * @param sample A train sample with image, current and target shape filled
        * @param meanShape Mean shape of the training samples 
        * @returns A boolean value: True denotes successful calculation of relative pixel else false
        */
        bool calcRelativePixelsParallel(trainSample& sample, vector<Point2f>& meanShape);



        // PREDICTION FUNCTIONS


        /** @brief This function traverse the cascade adding appropriate leaves on each regression tree in the current shape and yeilds the desired Landmarks
        * @param image A variable of cv::Mat containing the image whose landmarks have to be detected
        * @param faces A vector of type cv::Rect containing the faceDetector results
        * @param cascadeFinal A vector of vector of regressionTree loaded from a trained model
        * @param pixelCoordinates A vector of vector of type cv::Point2f in which the test co-ordinates are loaded
        * @returns A vector of vector of type cv::Point2f containing the desired Landmarks of all the faces in the image provided 
        */
        vector< vector<Point2f> > getFacialLandmarks(Mat image, vector<Rect>& faces, vector< vector<regressionTree> >& cascadeFinal, vector< vector<Point2f>>& pixelCoordinates);



        // MODEL READING - WRITING FUNCTIONS
        

        /** @brief This function initiates the model writing once the model has been trained
        * @param fs An output stream variable opened earlier and into which the results will be written
        * @param forest A vector of vector of regressionTree trained during the training procedure
        * @param pixelCoordinates A vector of vector of type cv::Point2f in which the test co-ordinates are stored
        */
        void writeModel(ofstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> > pixelCoordinates);

        /** @brief This function is called inside writeModel function to write each cascade
        * @param fs An output stream variable opened earlier and into which the results will be written
        * @param forest A vector of type regressionTree denoting the ensemble of regression trees at each level
        */
        void writeCascade( ofstream& fs, vector<regressionTree>& forest);

        /** @brief This function is called inside writeCascade function to write each Regression Tree
        * @param fs An output stream variable opened earlier and into which the results will be written
        * @param tree A regression Tree of a cascade
        */
        void writeTree( ofstream& fs, regressionTree& tree);

        /** @brief This function is called inside writeTree function to write each Leaf
        * @param fs An output stream variable opened earlier and into which the results will be written
        * @param forest A vector of vector of type cv::Point2f storing leaf values of the regression tree 
        */
        void writeLeaf(ofstream& fs, vector< vector<Point2f> >& leaves);

        /** @brief This function is called inside writeTree function to write each Split Feature
        * @param fs An output stream variable opened earlier and into which the results will be written
        * @param forest A vector of split Features storing all the splits of the regression tree
        */
        void writeSplit(ofstream& fs, vector<splitFeature>& split);

        /** @brief This function loads a previously trained model to make predictions
        * @param fs An input stream from which the model will be read
        * @param forest A vector of regression trees in which the model will be loaded
        * @param pixelCoordinates A vector of vector of tupe cv::Point2f in which the pixel coordinates used during training are loaded
        */
        bool loadTrainedModel(ifstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> >& pixelCoordinates);
        


        // ACCURACY VERIFICATION FUNCTION
        

        /** @brief Give the distance between the centre of both the eyes to normalize the error during accuracy calculation(This function is only valid for 68 landmarks)
        * @param currentShape A vector of cv::Point2f containing the current Shape of the sample
        * @returns A std::double denoting the distance between the eyes
        */
        double getInterocularDistance (vector<Point2f>& currentShape);


        // DISPLAY AND SAVING FUNCTIONS


        /** @brief Display the result of detection given a sample (Can be used for any number of Landmarks)
        * @param samples A train sample containing current Shape and image
        * @returns A boolean value: True denoting successfull depiction else false
        */
        bool displayresults(trainSample& samples);

        /** @brief Display the result of detection given a vector of training samples (Can be used for any number of Landmarks)
        * @param samples A vector of training samples containg the images and current shape
        * @returns A boolean value: True denoting successfull depiction else false
        */
        bool displayresultsdataset(vector<trainSample>& samples);
        
        /** @brief This function display the Landmarks during prediction.( Can be used for any number of Landmarks)
        * @param image A variable of cv::Mat containing the image for augmentation
        * @param resultPoints A vector of vector of cv::Point2f containing the results of Facial Landmark prediction
        */
        void display(Mat& image, vector< vector<Point2f>>& resultPoints);
        
        /** @brief Render the face onto the image given a single sample, Giving an outline just for display similar to dlib ( Works only for 68 Landmarks)
        * @param sample A train sample containg the image and current shape
        * @param color A variable of type cv::Scalar assigning the color of the lines(in BGR format)
        * @param thickness A variable of type int aserting the thickness of the lines
        */
        void renderDetections(trainSample& sample, Scalar color, int thickness);

        /** @brief Rendering functions for multiple faces( Works only for 68 Landmarks)
        * @param image A variable of type cv::Mat containing the values of the image whose facial landmarks has to be rendered
        * @param faces A vector of type cv::Rect containg the results of Face detector
        * @param results A vector of vector of cv::Point2f containing the results after Landmark Detection
        * @param color A variable of type cv::Scalar assigning the color of the lines(in BGR format)
        * @param thickness A variable of type int aserting the thickness of the lines
        */
        void renderDetectionsperframe(Mat& image, vector<Rect>& faces, vector< vector<Point2f>>& results, Scalar color, int thickness);

        /** @brief This function saves the samples with the landmark detections on the face into a file
        * @param samples A single training sample containig the image and current shape to be saved
        * @param no A variable of type std::int specifying the number to be added in the name while saving sample
        */
        void savesample(trainSample samples, int no);

        /** @brief This function rescales the data in the input domain from the default 460x460 scale
        * @param results A vector of vector of cv::Point2f containing the current shapes of all the facees in the image
        * @param scalex A std::float denoting the scaling factor along X-axis(cols)
        * @param scaley A std::float denoting the scaling factor along Y-axis(rows)
        */
        void rescaleData(vector< vector<Point2f>>& results, float scalex, float scaley);
        


        unsigned long numTestSplits;
        
        Point2f meanShapeReferencePoints[3];
        
        vector<Point2f> meanShape;
        
        vector<Point2f> meanShapeBounds;


        /** @brief Constructor for class KazemiFaceAlignImpl, setting the default values of essential parameters */
        
        KazemiFaceAlignImpl()
        {
            numFaces = 1;
            numLandmarks = 68;
            cascadeDepth = 10;
            treeDepth = 5;
            numTreesperCascade = 500;
            learningRate = 0.1;
            oversamplingAmount = 100;
            feature_pool_size = 400;
            numTestCoordinates = 500;
            lambda = 0.1;
            numTestSplits = 20;
            numFeature = 400;
            numSamples = 300;
        }

    private:
        // TRAINIG AND PREDICTION PARAMETERS
        
        int numFaces;
        int numLandmarks;
        unsigned long numSamples;
        vector< vector<Point2f> > initialShape;
        unsigned long cascadeDepth;
        unsigned long treeDepth;
        unsigned long numTreesperCascade;
        float learningRate;
        unsigned long oversamplingAmount;
        unsigned long feature_pool_size;
        float lambda;
        unsigned long numTestCoordinates;
        int numFeature;
};

}
#endif
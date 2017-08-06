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
    //! vector to store the pixel values at the desired annotations locations.
    vector<double> pixelValues;
};
class KazemiFaceAlignImpl
{
    public:
        //KazemiFaceAlignImpl();
        //virtual ~KazemiFaceAlignImpl();
        //*@Returns number of faces detected in the image */
        int getFacesNum() const {return numFaces;}
        // /*@Returns number of landmarks to be considered */
        int getLandmarksNum() const {return numLandmarks;}
        //@ Returns cascade Depth
        int getCascadeDepth() const {return cascadeDepth;}
        //@ Sets cascade's Depth
        void setCascadeDepth(unsigned int);
        //@ Returns Tree Depth
        int getTreeDepth() const {return treeDepth;}
        //@ Sets Regression Tree's Depth
        void setTreeDepth(unsigned int);
        //@ Returns the left of child the Regression Tree
        unsigned long leftChild (unsigned long idx);
        //@ Returns the right child of the Regression Tree
        unsigned long rightChild (unsigned long idx);
        /*@reads the file list(xml) created by imagelist_creator.cpp */
        bool readAnnotationList(vector<cv::String>& l, string annotation_path_prefix);
        /*@Parse the txt file to extract image and its annotations*/
        bool readtxt(vector<cv::String>& filepath, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix);
        //@ Applies Haar based facedetectorl
        vector<Rect> faceDetector(Mat image,CascadeClassifier& cascade);
        //@ return an image
        Mat getImage(string imgpath,string path_prefix);
        //@ Gives initial fiducial Points respective to the mean shape
        bool calcMeanShapeBounds();
        //@ Calculate distance between given pixel co-ordinates
        double getDistance(Point2f first , Point2f second);
        //@ Calculate different between the annotations
        bool calcDiff(vector<Point2f>& target, vector<Point2f>& current, vector<Point2f>& output);
        //A Calculate Sum of annotation vectors
        bool calcSum(vector<Point2f>& input1, vector<Point2f>& input2, vector<Point2f>& output);
        //@
        bool calcMul(vector<Point2f>& target, vector<Point2f>& current, vector<Point2f>& output);
        //Binary file writing functions
        void writeModel(ofstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> > pixelCoordinates);

        void writeCascade( ofstream& fs, vector<regressionTree>& forest);

        void writeTree( ofstream& fs, regressionTree& tree);

        void writeLeaf(ofstream& fs, vector< vector<Point2f> >& leaves);

        void writeSplit(ofstream& fs, vector<splitFeature>& split);

        bool displayresults(trainSample& samples);

        //@ Prediction functions
        bool loadTrainedModel(ifstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> >& pixelCoordinates);

        bool displayresults2(vector<trainSample>& samples);
        //@
        vector< vector<Point2f> > getFacialLandmarks(Mat& image, vector< vector<regressionTree> >& cascadeFinal, vector< vector<Point2f>>& pixelCoordinates, CascadeClassifier& cascade);

        // PASS SOME CONFIG FILE FOR ALL THE INITIAL PARAMETERS
        KazemiFaceAlignImpl()
        {
            numFaces = 1;
            numLandmarks = 68;
            cascadeDepth = 10;
            treeDepth = 5;
            numTreesperCascade = 500;
            learningRate = 0.1;
            oversamplingAmount = 20;
            feature_pool_size = 400;
            numTestCoordinates = 400;
            lambda = 0.1;
            numTestSplits = 20;
            numFeature = 400;
        }
        //@ Randomly Generates splits given a set of pixel co-ordinates
        splitFeature randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates);
        //@
        splitFeature splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long start ,
                                    unsigned long end, vector<Point2f>& sum, vector<Point2f>& leftSum, vector<Point2f>& rightSum);
        //@
        bool extractPixelValues(trainSample &sample ,vector<Point2f>& pixelCoordinates);
        //@
        regressionTree buildRegressionTree(vector<trainSample>& samples, vector<Point2f>& pixelCoordinates);
        //@
        vector<regressionTree> gradientBoosting(vector<trainSample>& samples, vector<Point2f>& pixelCoordinates);
        //@
        unsigned long partitionSamples(splitFeature split, vector<trainSample>& samples,
                                        unsigned long start, unsigned long end);
        //@Intitiates the training of Cascade
        bool trainCascade(std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix, CascadeClassifier& cascade, string outputName);
        //@
        unsigned int findNearestLandmark(Point2f& pixelValue);
        
        bool calcRelativePixels(vector<Point2f>& sample,vector<Point2f>& pixel_coordinates);
        //@
        bool generateTestCoordinates(vector< vector<Point2f> >& pixelCoordinates);
        
        Mat normalizing_tform(Rect& r);

        Mat unnormalizing_tform(Rect& r);
        
        unsigned long nearest_shape_point(Point2f& pt);
        
        void savesample(trainSample samples, int no);
        
        bool fillData(vector<trainSample>& samples,std::unordered_map<string, vector<Point2f>>& landmarks,
                                    string path_prefix, CascadeClassifier& cascade);
        bool scaleData( vector<Point2f>& trainlandmarks,Mat& trainimages ,Size s);
        
        bool readnewdataset(vector<cv::String>& l, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix);
        bool displayresultstarget2(vector<trainSample>& samples);

        double getInterocularDistance (vector<Point2f>& currentShape);
        void writeCascadexml( FileStorage& fs,vector<regressionTree>& forest);
        void writeTreexml( FileStorage& fs, regressionTree& tree,unsigned long treeNo);
        void writeLeafxml( FileStorage& fs, vector< vector<Point2f> >& leaves);
        void writeSplitxml(FileStorage& fs, vector<splitFeature>& split);
        unsigned long numTestSplits;
    private:
        int numFaces;
        int numLandmarks;
        vector<Point2f> meanShape;
        vector<Point2f> meanShapeBounds;
        Point2f meanShapeReferencePoints[3];
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
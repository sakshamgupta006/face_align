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
#include "objdetect.hpp"
#include <bits/stdc++.h>

using namespace std;

namespace cv{
class KazemiFaceAlignImpl : public KazemiFaceAlign
{
    public:
        KazemiFaceAlignImpl();
        virtual ~KazemiFaceAlignImpl();
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
        bool readtxt(vector<cv::String>& filepath, std::map<string, vector<Point2f>>& landmarks, string path_prefix);
        //@ Extracts Mean Shape from the given dataset
        bool extractMeanShape(std::map<string, vector<Point2f>>& landmarks, string path_prefix,CascadeClassifier& cascade);
        //@ Applies Haar based facedetectorl
        vector<Rect> faceDetector(Mat image,CascadeClassifier& cascade);
        //@ return an image
        Mat getImage(string imgpath,string path_prefix);
        //@ Gives initial fiducial Points respective to the mean shape
        bool getInitialShape(Mat& image, CascadeClassifier& cascade);
        //@ Reads MeanShape into a vector
        bool readMeanShape();
        //@ Calculate distance between given pixel co-ordinates
        double getDistance(Point2f first , Point2f second);

        // PASS SOME CONFIG FILE FOR ALL THE INITIAL PARAMETERS
        KazemiFaceAlignImpl()
        {
            readMeanShape();
            numFaces = 1;
            numLandmarks = 194;
            cascadeDepth = 10;
            treeDepth = 4;
            num_trees_per_cascade = 500;
            nu = 0.1;
            oversamplingAmount = 20;
            feature_pool_size = 400;
            lambda = 0.1;
            numTestSplits = 20;
            numFeature = 400;
        }

    protected:
        //@ Randomly Generates splits given a set of pixel co-ordinates
        splitFeature randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates);
        //@
        splitFeature splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long begin , unsigned long end);
        //@
        bool extractPixelValues(trainSample &sample , vector<Point2f> pixelCoordinates);
        //@
        regressionTree buildRegressionTree(vector<trainSample>& samples, vector<Point2f> pixelCoordinates);
        //@
        unsigned long partitionSamples(splitFeature split, vector<trainSample>& samples, unsigned long start, unsigned long end);
    
    private:
        int numFaces;
        int numLandmarks;
        vector<Point2f> meanShape;
        vector< vector<Point2f> > initialShape;
        unsigned int cascadeDepth;
        unsigned int treeDepth;
        unsigned int num_trees_per_cascade;
        float nu;
        unsigned long oversamplingAmount;
        unsigned int feature_pool_size;
        float lambda;
        unsigned int numTestSplits;
        int numFeature;
};
}
#endif

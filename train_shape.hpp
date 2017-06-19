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

using namespace std;

namespace cv{
/** @brief Implementation of face alignnment from the paper:
 *  <a href="https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf">Vahid Kazemi and Josephine Sullivan, "One Millisecond Face Alignment with an Ensemble of Regression Trees."
 *  Computer Vision and Pattern Recognition, 2014. CVPR 2014.</a>
 */
class CV_EXPORTS KazemiFaceAligninclude
{
public:
    //@ Returns the left of child the Regression Tree
    virtual unsigned long leftChild (unsigned long idx);
    //@ Returns the right child of the Regression Tree
    virtual unsigned long rightChild (unsigned long idx);
    /*@reads the file list(xml) created by imagelist_creator.cpp */
    virtual bool readAnnotationList(vector<cv::String>& l, string annotation_path_prefix);
    /*@Parse the txt file to extract image and its annotations*/
    virtual bool readtxt(vector<cv::String>& filepath, std::map<string, vector<Point2f>>& landmarks, string path_prefix);
    //@ Extracts Mean Shape from the given dataset
    virtual bool extractMeanShape(std::map<string, vector<Point2f>>& landmarks, string path_prefix,CascadeClassifier& cascade);
    //@ Applies Haar based facedetectorl
    virtual vector<Rect> faceDetector(Mat image,CascadeClassifier& cascade);
    //@ return an image
    virtual Mat getImage(string imgpath,string path_prefix);
    //@ Gives initial fiducial Points respective to the mean shape
    virtual bool getInitialShape(Mat& image, CascadeClassifier& cascade);
    //@ Reads MeanShape into a vector
    virtual bool readMeanShape();
    //@ Calculate distance between given pixel co-ordinates
    virtual double getDistance(Point2f first , Point2f second);
};
CV_EXPORTS Ptr<KazemiFaceAligninclude> create();
}
#endif
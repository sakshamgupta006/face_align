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
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "train_shape.hpp"
#include "opencv2/videoio.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

static void help()
{
    cout << "To be written near code completion"<<endl;
}

int main(int argc, const char** argv)
{
    string cascadeName, inputName;
    CascadeClassifier cascade;
    string poseTree;
    cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade | /home/cooper/gsoc/opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"  //Add LBP , HOG and HAAR based detectors also
            "{path | ../data/300wcropped/ | }"
            "{poseTree| 194_landmarks_face_align.dat |}"
            "{@filename| ../data/300wcropped/outdoor_292.png |}"
        );
    if(parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    poseTree = parser.get<string>("poseTree");
    inputName = parser.get<string>("@filename");  // Add multiple file support
    Mat image = imread(inputName);
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    KazemiFaceAlignImpl predict;
    ifstream fs(poseTree, ios::binary);
    if(!fs.is_open())
    {
        cerr << "Failed to open trained model file " << poseTree << endl;
        help();
        return 1;
    }
    vector< vector<regressionTree> > forests;
    vector< vector<Point2f> > pixelCoordinates;
    predict.loadTrainedModel(fs, forests, pixelCoordinates);
    predict.calcMeanShapeBounds();
    cout<<"Model Loaded"<<endl;
    vector< vector<Point2f> > resultLandmarks;
    //VideoCapture cap(0);
    // while(1)
    // {   
       // cap >> image;
        predict.getFacialLandmarks(image, forests, pixelCoordinates, cascade);
      //  waitKey(0) ;//>= 0) break;
    //}
    return 0;
}
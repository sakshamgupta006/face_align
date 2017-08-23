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
#include "../include/train_shape.hpp"
#include "opencv2/videoio.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the training of Face Alignment Technique by Vahid Kazemi.\nThis approach works with LBP, HOG and HAAR based face detectors.\n"
            "This module can work on any number of landmarks on face. With some modififcation it can be used for Landmark detection of any shape\n"
            "Usage:\n"
            "./Train_ex [-cascade=<face detector model>(optional)this is the cascade mlodule used for face detection]\n"
               "   [-path=<path to dataset> specifies the path to dataset on which the model will be trained]]\n"
               "   [-landmarks=<Number of Landmarks in the Dataset>(optional)The iBug 300w has 68 landmarks by default]\n"
               "   [-oversampling=<Number of different Initializations of each Image>(optional)(20 default)]\n"
               "   [-learningrate=<Learning Rate during Regression>(optional)(0.1 by default)]\n"
               "   [-cascadedepth=<Number of Cascade's>(optional)(10 by default)]\n"
               "   [-treedepth=<Depth of each Regression Tree>(optional)(5 by default)]\n"
               "   [-treespercascade=<Number of Tree's in each cascade>(optional)(500 by default)]\n"
               "   [-testcoordinates=<Number of test coordinates>(optional)(500 by default)]\n"
               "   [-lambda=<Priori for Randonm Feature Selection>(optional)(0.1 by default)]\n"
               "   [-samples=<Number of images to be trained on from dataset>(optional)(300 by default)]"
               "   [@filename(Output Model Filename)(\"68_landmarks_face_align.dat\" by default)]\n\n"
            "for one call:\n"
            "./Train_ex -cascade=\"../data/haarcascade_frontalface_alt2.xml\" -path=\"../data/dataset/\" -landmarks=68 -oversampling=20 -learningrate=0.1 -cascadedepth=10 -treedepth=5 -treespercascade=500 -testcoordinates=500 -lambda=0.1 -samples=300 68_landmarks_face_align.dat\n"
            "Using OpenCV version " << CV_VERSION << "\n" << endl;

}

Rect calctightbound(Mat image)
{
    cvtColor(image,image,CV_BGR2GRAY);
    int x1=0,x2=0,y1=0,y2=0;
    for (int i = 0; i < image.cols; ++i)
    {
        for (int j = 0; j < image.rows; ++j)
        {
            if(image.at<uchar>(j,i) > 0 )
            {
                y1 = i;
                break;
            }
        }
    }
    for (int i = image.cols-1; i >= 0 ; i--)
    {
        for (int j = 0; j < image.rows; ++j)
        {
            if(image.at<uchar>(j,i) > 0)
            {
                y2 = i;
                break;
            }
        }
    }
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            if(image.at<uchar>(i,j) > 0)
            {
                x1 = i;
                break;
            }
        }
    }
    for (int i = image.rows-1; i >= 0; i--)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            if(image.at<uchar>(i,j) > 0)
            {
                x2 = i;
                break;
            }
        }
    }
    // cout<<x1<<" "<<x2<<endl;
    // cout<<y1<<" "<<y2<<endl;
    // circle(image, Point(x1,y1),5, Scalar(0,0,255), -1);
    // circle(image, Point(x1,y2),5, Scalar(0,0,255), -1);
    // circle(image, Point(x2,y1),5, Scalar(0,0,255), -1);
    // circle(image, Point(x2,y2),5, Scalar(0,0,255), -1);
    // imshow("show", image);
    
Rect r;
r.y = min(x1,x2);
r.x = min(y1,y2);
r.width = abs(y2 - y1);
r.height = abs(x2 - x1);
return r;
}


void overlayImage(const cv::Mat &background, const cv::Mat &foreground, 
  cv::Mat &output, cv::Point2i location)
{
  background.copyTo(output);


  // start at the row indicated by location, or at row 0 if location.y is negative.
  for(int y = std::max(location.y , 0); y < background.rows; ++y)
  {
    int fY = y - location.y; // because of the translation

    // we are done of we have processed all rows of the foreground image.
    if(fY >= foreground.rows)
      break;

    // start at the column indicated by location, 

    // or at column 0 if location.x is negative.
    for(int x = std::max(location.x, 0); x < background.cols; ++x)
    {
      int fX = x - location.x; // because of the translation.

      // we are done with this row if the column is outside of the foreground image.
      if(fX >= foreground.cols)
        break;

      // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
      double opacity =
        ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

        / 255.;


      // and now combine the background and foreground pixel, using the opacity, 

      // but only if opacity > 0.
      for(int c = 0; opacity > 0 && c < output.channels(); ++c)
      {
        unsigned char foregroundPx =
          foreground.data[fY * foreground.step + fX * foreground.channels() + c];
        unsigned char backgroundPx =
          background.data[y * background.step + x * background.channels() + c];
        output.data[y*output.step + output.channels()*x + c] =
          backgroundPx * (1.-opacity) + foregroundPx * opacity;
      }
    }
  }
}

int main(int argc, const char** argv)
{
    VideoCapture capture;
    string cascadeName, inputName;
    CascadeClassifier cascade;
    string poseTree;
    Mat image, frame;
    cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade | ../data/haarcascade_frontalface_alt2.xml|}"  //Only HAAR based detectors
            "{model| ../data/68_landmarks_face_align.dat |}"        //will work as the model is
            "{@filename||}"                                         //trained using HAAR.
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
    poseTree = parser.get<string>("model");
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
    vector< vector<Point2f>> result;
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(camera))
            cout << "Capture from camera #" <<  camera << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            if(!capture.open( inputName ))
                cout << "Could not read " << inputName << endl;
        }
    }
    else
    {
        image = imread( "../data/lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
    }
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        Mat spec = imread("../data/Glasses.png",-1);
        Rect tight;
        Mat img2 = imread("../data/clooney.jpg");
        resize(img2, img2, Size(460,460));
        vector<Rect> faces2 = predict.faceDetector(img2, cascade);
        vector<vector<Point2f>> res2;
        res2 = predict.getFacialLandmarks(img2, faces2, forests, pixelCoordinates);
        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;
            resize(frame, frame, Size(460,460));
            vector<Rect> faces  = predict.faceDetector(frame, cascade);
            if(faces.size() == 0)
            	continue;
            result = predict.getFacialLandmarks(frame, faces, forests, pixelCoordinates);
            //predict.renderDetectionsperframe(frame, faces, result, Scalar(0,255,0), 2);
            Point2f source[3], dst[3];
            source[0] = res2[0][0]; source[1] = res2[0][16]; source[2] = res2[0][8];
            dst[0] = result[0][0]; dst[1] = result[0][16]; dst[2] = result[0][8];
            Mat warp_mat = getAffineTransform(source, dst);
           	Mat spec_copy = spec.clone();
           	warpAffine(spec_copy, spec_copy, warp_mat, spec_copy.size());
            tight = calctightbound(spec);			
            Mat spec_cropped = spec(tight);
            overlayImage(frame, spec_cropped, frame, result[0][0]);
           	imshow("Filter", frame);
            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else
    {
        cout << "Detecting landmarks in " << inputName << endl;
        if( !image.empty() )
        {
            resize(image, image, Size(460,460));
            vector<Rect> faces = predict.faceDetector(image, cascade);
            result = predict.getFacialLandmarks(image, faces, forests, pixelCoordinates);
            predict.renderDetectionsperframe(image, faces, result, Scalar(0, 255, 0), 2);
            waitKey(0);
            Mat img2 = imread("../data/clooney.jpg");
            resize(img2, img2, Size(460,460));
            vector<Rect> faces2 = predict.faceDetector(img2, cascade);
            vector<vector<Point2f>> res2;
            res2 = predict.getFacialLandmarks(img2, faces2, forests, pixelCoordinates);
            predict.renderDetectionsperframe(img2, faces2, res2, Scalar(0, 255, 0), 2);
            waitKey(0);
            Point2f source[3], dst[3];
            source[0] = res2[0][0]; source[1] = res2[0][16]; source[2] = res2[0][8];
            dst[0] = result[0][0]; dst[1] = result[0][16]; dst[2] = result[0][8];
            Mat warp_mat = getAffineTransform(source, dst);
            Mat spec = imread("../data/Glasses.png",-1);
           	warpAffine(spec, spec, warp_mat, spec.size());
            Rect tight;
            tight = calctightbound(spec);			
            cout<<tight<<endl;
            Mat spec_cropped = spec(tight);
            overlayImage(image, spec_cropped, image, result[0][17]);
           	imshow("Filter", image);
           	waitKey(0);           	
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        resize(image, image, Size(460,460));
                        vector<Rect> faces = predict.faceDetector(image, cascade);
                        result = predict.getFacialLandmarks(image, faces, forests, pixelCoordinates);
                        predict.renderDetectionsperframe(image, faces, result, Scalar(0,255,0), 2);
                        char c = (char)waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }
    return 0;
}
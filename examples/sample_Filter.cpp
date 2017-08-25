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
    cout << "\nThis program demonstrates Snapchat Filter like Application using Alignment Technique by Vahid Kazemi.\nThis application supports three filter as of now.\n"
            "The application can be customized to support any face filtes.\n"
            "Any new filter must be first mannually fitted over clooney.jpg in data folder for best results.\n"
            "Usage:\n"
            "./Filter_ex [-cascade=<face detector model>(optional)this is the cascade mlodule used for face detection]\n"
               "   [-model=<path to trained model> specifies the path to trained model]]\n"
               "   [-filter=<Give different numbers for different Filters>(1: Glasses , 2: Batman , 3: Chetah)\n"
               "   [@filename(For image: provide path to image, For multiple images: Provide a txt file with path to images, For video input: Provide path to video, For live input: Leave blank)]\n\n"
            "for one call:\n"
            "./Filter_ex -cascade=\"../data/haarcascade_frontalface_alt2.xml\" -model=\"../data/68_landmarks_face_align.dat/\" -filter=1 image1.png\n"
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
Rect r;
r.y = min(x1,x2);
r.x = min(y1,y2);
r.width = abs(y2 - y1);
r.height = abs(x2 - x1);
return r;
}

//This approach is used to overlay a png image over another image removing the background of overlayed image
void overlayImage(const cv::Mat &background, const cv::Mat &foreground, cv::Mat &output, cv::Point2i location)
{
  background.copyTo(output);
  for(int y = std::max(location.y , 0); y < background.rows; ++y)
  {
    int fY = y - location.y;
    if(fY >= foreground.rows)
      break;
    for(int x = std::max(location.x, 0); x < background.cols; ++x)
    {
      int fX = x - location.x;
      if(fX >= foreground.cols)
        break;
      double opacity =((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])/ 255.;
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
            "{cascade | ../data/haarcascade_frontalface_alt2.xml|}"  
            "{model| ../data/68_landmarks_face_align.dat |}"        
            "{filter | 1 |}"
            "{@filename||}"
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
    KazemiFaceAlignImpl filter;
    ifstream fs(poseTree, ios::binary);
    if(!fs.is_open())
    {
        cerr << "Failed to open trained model file " << poseTree << endl;
        help();
        return 1;
    }
    vector< vector<regressionTree> > forests;
    vector< vector<Point2f> > pixelCoordinates;
    filter.loadTrainedModel(fs, forests, pixelCoordinates);
    filter.calcMeanShapeBounds();
    vector< vector<Point2f>> result;
    unsigned long filterval = parser.get<unsigned long>("filter");
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

    Mat spec;
    if(filterval == 1)
        spec = imread("../data/Glasses.png",-1);
    else if(filterval == 2)
        spec = imread("../data/batman.png", -1);
    else if(filterval == 3)
        spec = imread("../data/leopard.png", -1);
    Rect tight;
    Mat img2 = imread("../data/clonney.png");
    vector<Rect> faces2 = filter.faceDetector(img2, cascade);
    vector<vector<Point2f>> res2;
    res2 = filter.getFacialLandmarks(img2, faces2, forests, pixelCoordinates);
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;
            resize(frame, frame, Size(460,460));
            vector<Rect> faces  = filter.faceDetector(frame, cascade);
            if(faces.size() == 0)
            	continue;
            result = filter.getFacialLandmarks(frame, faces, forests, pixelCoordinates);
            //filter.renderDetectionsperframe(frame, faces, result, Scalar(0,255,0), 2);
            Point2f source[3], dst[3];
            source[0] = res2[0][0]; source[1] = res2[0][16]; source[2] = res2[0][8];
            dst[0] = result[0][0]; dst[1] = result[0][16]; dst[2] = result[0][8];
            Mat warp_mat = getAffineTransform(source, dst);
           	Mat spec_copy = spec.clone();
           	warpAffine(spec_copy, spec_copy, warp_mat, spec_copy.size());
            tight = calctightbound(spec);			
            Mat spec_cropped = spec(tight);
            switch(filterval)
            {
                case 1:
                    overlayImage(frame, spec_cropped, frame, result[0][17]);
                    break;
                case 2:
                    overlayImage(frame, spec_cropped, frame, result[0][17]);
                    break;
                case 3:
                    overlayImage(frame, spec_cropped, frame, result[0][17]);
                    break;    
            }
           	imshow("Filter", frame);
            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else
    {
        if( !image.empty() )
        {
            resize(image, image, Size(460,460));
            vector<Rect> faces = filter.faceDetector(image, cascade);
            if(faces.empty())
            {
                cerr << "Aw..cannot find any faces in the image.." << endl;
                return -1;
            }
            result = filter.getFacialLandmarks(image, faces, forests, pixelCoordinates);
            filter.renderDetectionsperframe(image, faces, result, Scalar(0,255,0), 2);
            Point2f source[3], dst[3];
            source[0] = res2[0][0]; source[1] = res2[0][16]; source[2] = res2[0][8];
            dst[0] = result[0][0]; dst[1] = result[0][16]; dst[2] = result[0][8];
            Mat warp_mat = getAffineTransform(source, dst);
            warpAffine(spec, spec, warp_mat, spec.size());
            Rect tight;
            tight = calctightbound(spec);			
            Mat spec_cropped = spec(tight);
            switch(filterval)
            {
                case 1:
                    overlayImage(image, spec_cropped, image, result[0][17]);
                    break;
                case 2:
                    overlayImage(image, spec_cropped, image, result[0][17]);
                    break;
                case 3:
                    
                    overlayImage(image, spec_cropped, image, result[0][17]);
                    break;    
            }
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
                        vector<Rect> faces = filter.faceDetector(image, cascade);
                        result = filter.getFacialLandmarks(image, faces, forests, pixelCoordinates);
                        filter.renderDetectionsperframe(image, faces, result, Scalar(0,255,0), 2);
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
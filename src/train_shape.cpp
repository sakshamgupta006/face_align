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
#include <vector>
#include <climits>
#include <iostream>

using namespace std;
using namespace cv;

namespace cv
{
void KazemiFaceAlignImpl::setnumLandmarks(unsigned long numberLandmarks)
{
    if(numberLandmarks < 0)
    {
        String errmsg = "Invalid Number of Landmarks";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    numLandmarks = numberLandmarks;
}

void KazemiFaceAlignImpl::setOverSampling(unsigned long oversampling)
{
    if(oversampling <= 0)
    {
        String errmsg = "Invalid Over Sampling Amount";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    oversamplingAmount = oversampling;
}

void KazemiFaceAlignImpl::setLearningRate(float learningrate)
{
    if(learningrate < 0)
    {
        String errmsg = "Invalid Learing Rate";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    learningRate = learningrate;
}

void KazemiFaceAlignImpl::setCascadeDepth(unsigned long newdepth)
{
    if(newdepth < 0)
    {
        String errmsg = "Invalid Cascade Depth";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    cascadeDepth = newdepth;
}


void KazemiFaceAlignImpl::setTreeDepth(unsigned long newdepth)
{
    if(newdepth < 0)
    {
        String errmsg = "Invalid Tree Depth";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    treeDepth = newdepth;
}

void KazemiFaceAlignImpl::setTreesPerCascade(unsigned long treespercascade)
{
    if(treespercascade < 0)
    {
        String errmsg = "Invalid Number of Trees per Cascade";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    numTreesperCascade = treespercascade;
}

void KazemiFaceAlignImpl::setTestCoordinates(unsigned long newcoordinates)
{
    if(newcoordinates < 0)
    {
        String errmsg = "Invalid number of Test coordinates";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    numTestCoordinates = newcoordinates;
}

void KazemiFaceAlignImpl::setTestSplits(unsigned long testsplits)
{
    if(testsplits < 0)
    {
        String errmsg = "Invalid number of Test Splits";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    numTestSplits = testsplits;
}

void KazemiFaceAlignImpl::setLambda(float Lambda)
{
    if(Lambda < 0)
    {
        String errmsg = "Invalid Lambda";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    lambda = Lambda;
}

void KazemiFaceAlignImpl::setnumSamples(unsigned long numsamples)
{
    if(numSamples < 5)
    {
        String errmsg = "Invalid number of Samples";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    numSamples = numsamples;
}

unsigned long KazemiFaceAlignImpl::leftChild(unsigned long idx)
{
    return 2*idx + 1;
}

unsigned long KazemiFaceAlignImpl::rightChild(unsigned long idx)
{
    return 2*idx + 2;
}

//to read the annotation files file of the annotation files
bool KazemiFaceAlignImpl::readAnnotationList(vector<cv::String>& l, string annotation_path_prefix )
{
    string annotationPath = annotation_path_prefix + "*.txt";
    glob(annotationPath,l,false);
    return true;
}

bool KazemiFaceAlignImpl::readnewdataset(vector<cv::String>& l, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix)
{
    vector<cv::String> filenames;
    vector<cv::String> files_png;
    string annotationPath = path_prefix + "*.png";
    string annotationPath3 = path_prefix + "*.jpg";
    string annotationPath2 = path_prefix + "*.pts";
    glob(annotationPath3,l,false);
    glob(annotationPath2,filenames,false);
    glob(annotationPath,files_png,false);
    for (int i = 0; i < files_png.size(); ++i)
    {
        l.push_back(files_png[i]);
    }
    vector<Point2f> temp;
    string s, tok, randomstring;
    vector<string> coordinates;
    ifstream f;
    for(unsigned long j = 0; j < filenames.size(); j++)
    {
        f.open(filenames[j].c_str(),ios::in);
        if(!f.is_open())
        {
            CV_Error(Error::StsError, "File cannot be opened");
            return false;
        }
        getline(f,randomstring);
        getline(f,randomstring);
        getline(f,randomstring);
        for(int i = 0; i < 68; i++)
        {
            Point2f point;
            getline(f,s);
            stringstream ss(s);
            while(getline(ss, tok,' ')) 
            {
                coordinates.push_back(tok);
                tok.clear();
            }
            point.x = (float)atof(coordinates[0].c_str());
            point.y = (float)atof(coordinates[1].c_str());
            coordinates.clear();
            temp.push_back(point);
        }
        string v = l[j];
        landmarks[v] = temp;
        temp.clear();
        f.close();
    }
    return true;
}

bool KazemiFaceAlignImpl::readmirror(vector<cv::String>& l, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix)
{
    vector<cv::String> filenames;
    vector<cv::String> files_png;
    string annotationPath3 = path_prefix + "*_mirror.jpg";
    string annotationPath2 = path_prefix + "*.pts";
    glob(annotationPath3,files_png,false);        
    vector<Point2f> temp, temp2;
    string s, tok, randomstring;
    vector<string> coordinates;
    ifstream f;

    for(unsigned long j = 0; j < files_png.size(); j++)
    {
        Mat frame = imread(files_png[j]);
        string pts = files_png[j];
        String mirrorstringfile = "_mirror.jpg";
        size_t i1 = pts.find(mirrorstringfile);
        if (i1 != std::string::npos)
            pts.erase(i1,mirrorstringfile.length());
        pts = pts + ".pts";
        f.open(pts.c_str(),ios::in);
        if(!f.is_open())
        {
            CV_Error(Error::StsError, "File cannot be opened");
            return false;
        }
        getline(f,randomstring);
        getline(f,randomstring);
        getline(f,randomstring);
        for(int i = 0; i < 68; i++)
        {
            Point2f point, point2;
            getline(f,s);
            stringstream ss(s);
            while(getline(ss, tok,' ')) 
            {
                coordinates.push_back(tok);
                tok.clear();
            }
            point.x = (float)atof(coordinates[0].c_str());
            point.y = (float)atof(coordinates[1].c_str());
            point2.x = (float)abs( frame.cols - point.x);
            point2.y = point.y;
            coordinates.clear();
            temp.push_back(point);
            temp2.push_back(point2);
        }
        string img = files_png[j];
        String mirrorstring = "_mirror";
        size_t i = files_png[j].find(mirrorstring);
        if (i != std::string::npos)
            img.erase(i,mirrorstring.length());
        string v = img;
        landmarks[v] = temp;
        v = files_png[j];
        landmarks[v] = temp2;
        temp.clear();
        temp2.clear();
        f.close();
    }
    return true;
}



//read txt files iteratively opening image and its annotations
bool KazemiFaceAlignImpl::readtxt(vector<cv::String>& filepath, std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix)
{
    //txt file read initiated
    vector<cv::String>::iterator fileiterator = filepath.begin();
    for (; fileiterator != filepath.end() ; fileiterator++)
    {
        ifstream file;
        file.open((string)*fileiterator);
        string key,line;
        getline(file,key);
        key.erase(key.length()-1);
        vector<Point2f> landmarks_temp;
        while(getline(file,line))
        {
            stringstream linestream(line);
            string token;
            vector<string> location;
            while(getline(linestream, token,','))
            {
                location.push_back(token);
            }
            landmarks_temp.push_back(Point2f((float)atof(location[0].c_str()),(float)atof(location[1].c_str())));
        }
        file.close();
        landmarks[key] = landmarks_temp;
        //file reading completed
    }
    return true;
}

vector<Rect> KazemiFaceAlignImpl::faceDetector(Mat image,CascadeClassifier& cascade)
{
    vector<Rect> faces;
    int scale = 1;
    Mat gray;
    cvtColor( image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray,gray);
    cascade.detectMultiScale(gray, faces, 1.1, 3, 0 |CASCADE_SCALE_IMAGE,Size(100, 100));
    numFaces = faces.size();
    for(unsigned long i = 0; i < faces.size(); i++)
    {
        faces[i].width = faces[i].width;
        faces[i].height = 1.2*faces[i].height;
    }
    return faces;
}


bool KazemiFaceAlignImpl::calcDiff(vector<Point2f>& input1, vector<Point2f>& input2, vector<Point2f>& output)
{
    output.resize(input1.size());
    for (unsigned long i = 0; i < input1.size(); ++i)
    {
        output[i] = input1[i] - input2[i];
    }
    return true;
}

bool KazemiFaceAlignImpl::calcSum(vector<Point2f>& input1, vector<Point2f>& input2, vector<Point2f>& output)
{
    output.resize(input1.size());
    for (unsigned long i = 0; i < input1.size(); ++i)
    {
        output[i] = input1[i] + input2[i];
    }
    return true;
}

bool KazemiFaceAlignImpl::calcMeanShapeBounds()
{
    double meanShapeRectminx, meanShapeRectminy, meanShapeRectmaxx, meanShapeRectmaxy;
    double meanX[meanShape.size()] , meanY[meanShape.size()];
    int pointcount=0;
    for (vector<Point2f>::iterator it = meanShape.begin(); it != meanShape.end(); ++it)
    {
        meanX[pointcount] = (*it).x;
        meanY[pointcount] = (*it).y;
        pointcount++;
    }
    meanShapeRectminx = *min_element(meanX , meanX + meanShape.size());
    meanShapeRectmaxx = *max_element(meanX , meanX + meanShape.size());
    meanShapeRectminy = *min_element(meanY , meanY + meanShape.size());
    meanShapeRectmaxy = *max_element(meanY , meanY + meanShape.size());
    meanShapeBounds.push_back(Point2f(meanShapeRectminx, meanShapeRectminy));
    meanShapeBounds.push_back(Point2f(meanShapeRectmaxx, meanShapeRectmaxy));
    meanShapeReferencePoints[0] = Point2f(meanShapeRectminx, meanShapeRectminy);
    meanShapeReferencePoints[1] = Point2f(meanShapeRectmaxx, meanShapeRectminy);
    meanShapeReferencePoints[2] = Point2f(meanShapeRectminx, meanShapeRectmaxy);
    return true;
}

void KazemiFaceAlignImpl::renderDetections(trainSample& sample, Scalar color, int thickness)
{
    Mat image = sample.img.clone();
    vector<Point2f> temp1(sample.currentShape.size());
    Mat unorm_tform  = unnormalizing_tform(sample.rect[0]);
    for (unsigned long j = 0; j < sample.currentShape.size(); ++j)
    {
        Mat temp = (Mat_<double>(3,1)<< sample.currentShape[j].x , sample.currentShape[j].y , 1);
        Mat res = unorm_tform * temp;
        temp1[j].x = res.at<double>(0,0);
        temp1[j].y = res.at<double>(1,0);
    }
    //Chin
    for(unsigned long i = 1; i <= 16; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    //Top of nose
    for(unsigned long i = 28; i <= 30; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    //Left Eyebrow
    for(unsigned long i = 18; i <= 21; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    //Right Eyebrow
    for(unsigned long i = 23; i <= 26; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    //Bottom Part of nose
    for(unsigned long i = 31; i <= 35; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    //Nose to bottom part above
    line(image, temp1[30], temp1[35], color, thickness, CV_AA);
    //Left Eye
    for(unsigned long i = 37; i <= 41; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    line(image, temp1[36], temp1[41], color, thickness, CV_AA);
    //Right Eye
    for(unsigned long i = 43; i <= 47; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    line(image, temp1[42], temp1[47], color, thickness, CV_AA);
    //Lips outer part
    for(unsigned long i = 49; i <= 59; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    line(image, temp1[48], temp1[59], color, thickness, CV_AA);
    //Lips inside part
    for(unsigned long i = 61; i <= 67; i++)
        line(image, temp1[i], temp1[i-1], color, thickness, CV_AA);
    line(image, temp1[60], temp1[67], color, thickness, CV_AA);

    imshow("Rendered Image", image);
    waitKey(0);
}

void KazemiFaceAlignImpl::renderDetectionsperframe(Mat& image, vector<Rect>& faces, vector< vector<Point2f>>& results, Scalar color, int thickness)
{
    for (unsigned long k = 0; k < faces.size(); ++k)
    {
        //Chin
        for(unsigned long i = 1; i <= 16; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        //Top of nose
        for(unsigned long i = 28; i <= 30; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        //Left Eyebrow
        for(unsigned long i = 18; i <= 21; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        //Right Eyebrow
        for(unsigned long i = 23; i <= 26; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        //Bottom Part of nose
        for(unsigned long i = 31; i <= 35; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        //Nose to bottom part above
        line(image, results[k][30], results[k][35], color, thickness, CV_AA);
        //Left Eye
        for(unsigned long i = 37; i <= 41; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        line(image, results[k][36], results[k][41], color, thickness, CV_AA);
        //Right Eye
        for(unsigned long i = 43; i <= 47; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        line(image, results[k][42], results[k][47], color, thickness, CV_AA);
        //Lips outer part
        for(unsigned long i = 49; i <= 59; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        line(image, results[k][48], results[k][59], color, thickness, CV_AA);
        //Lips inside part
        for(unsigned long i = 61; i <= 67; i++)
            line(image, results[k][i], results[k][i-1], color, thickness, CV_AA);
        line(image, results[k][60], results[k][67], color, thickness, CV_AA);
    }
    imshow("Rendered Results", image);
}
}
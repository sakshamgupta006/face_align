/*By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.*/
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "train_shape.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

namespace cv
{

void KazemiFaceAlignImpl::setCascadeDepth(unsigned int newdepth)
{
    if(newdepth < 0)
    {
        String errmsg = "Invalid Cascade Depth";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    cascadeDepth = newdepth;
}

//KazemiFaceAlignImpl::~KazemiFaceAlignImpl()

void KazemiFaceAlignImpl::setTreeDepth(unsigned int newdepth)
{
    if(newdepth < 0)
    {
        String errmsg = "Invalid Tree Depth";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    treeDepth = newdepth;
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

bool KazemiFaceAlignImpl::readMeanShape()
{
    string meanShapefile = "meanshape.txt";
    ifstream f(meanShapefile.c_str());
    string line;
    getline(f,line);
    while(getline(f,line))
    {
        stringstream linestream(line);
        string token;
        vector<string> location;
        while(getline(linestream, token,','))
        {
            location.push_back(token);
        }
        meanShape.push_back(Point2f((float)atof(location[0].c_str()),(float)atof(location[1].c_str())));
    }
    calcMeanShapeBounds();
    cout<<"MeanShape Loaded and Bounds calculated"<<endl;
    return true;
}

vector<Rect> KazemiFaceAlignImpl::faceDetector(Mat image,CascadeClassifier& cascade)
{
    vector<Rect> faces;
    int scale = 1;
    //remove once testing is complete
    Mat gray;
    cvtColor( image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray,gray);
    cascade.detectMultiScale( gray, faces,
        1.1, 3, 0
        //|CASCADE_FIND_BIGGEST_OBJECT,
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(100, 100) );
    numFaces = faces.size();
    for (int i = 0; i < numFaces; ++i)
    {
        faces[i].width += image.rows/20;
        faces[i].height += image.cols/10;
    }
    // for ( size_t i = 0; i < faces.size(); i++ )
    // {
    //     Rect r = faces[i];
    //     Scalar color = Scalar(255,0,0);
    //     double aspect_ratio = (double)r.width/r.height;
    //     rectangle( image, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
    //                    cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
    //                    color, 3, 8, 0);
    // }
    return faces;
}

Mat KazemiFaceAlignImpl::getImage(string imgpath, string path_prefix)
{
    return imread(path_prefix + imgpath + ".jpg");
}

bool KazemiFaceAlignImpl::extractMeanShape(std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix,CascadeClassifier& cascade)
{
    //string meanShapeFileName = "mean_shape.xml";
    std::unordered_map<string, vector<Point2f>>::iterator dbIterator = landmarks.begin(); // random inititalization as
    //any face file can be taken by imagelist creator
    //find the face size dimmensions which will be used to center and scale each database image
    Mat initialImage = getImage(dbIterator->first,path_prefix);
    if(initialImage.empty())
    {
        cerr<<"ERROR: Image not loaded...Directory not found!!"<<endl;
        return 0;
    }
    //apply face detector on the image
    //assuming that the intitial shape contains only a single face for now
    vector<Rect> initial = faceDetector(initialImage,cascade);
    //now we have the initial/source image
    Point2f refrencePoints[3];
    refrencePoints[0] = Point2f(initial[0].x,initial[0].y);
    refrencePoints[1] = Point2f((initial[0].x + initial[0].width),initial[0].y);
    refrencePoints[2] = Point2f(initial[0].x , (initial[0].y + initial[0].height));
    //Calculate Face Rectangle and Affine Matrix for the whole dataset
    meanShape.clear();
    int count =0;
    Point2f mean[numLandmarks];
    //Enter the value of intital shape in mean shape
    dbIterator++;
    for (; dbIterator != landmarks.end() ; dbIterator++)
    {
        cout<<count++<<endl;
        Mat currentImage = getImage(dbIterator->first,path_prefix);
        if(currentImage.empty())
        {
            cerr<<"ERROR: Image not loaded...Directory not found!!"<<endl;
            break;
        }
        vector<Rect> currentFaces = faceDetector(currentImage,cascade);
        if(currentFaces.empty())
        {
            cerr<<"No faces found skipping the image"<<endl;
            continue;
        }
        Point2f currentPoints[3];
        currentPoints[0] = Point2f(currentFaces[0].x,currentFaces[0].y);
        currentPoints[1] = Point2f((currentFaces[0].x + currentFaces[0].width),currentFaces[0].y);
        currentPoints[2] = Point2f(currentFaces[0].x , (currentFaces[0].y + currentFaces[0].height));
        Mat affineMatrix = getAffineTransform(currentPoints , refrencePoints);
        int landmarkIterator = 0;
        //Transform each fiducial Point to get it relative to the initital image
        for (vector< Point2f >::iterator fiducialIt = dbIterator->second.begin() ; fiducialIt != dbIterator->second.end() ; fiducialIt++ )
        {
            Mat fiducialPointMatrix = (Mat_<double>(3,1) << (*fiducialIt).x, (*fiducialIt).y , 1);
            Mat resultAffineMatrix = (Mat_<double>(3,1)<<0,0,1);
            resultAffineMatrix = affineMatrix*fiducialPointMatrix;
            //warpAffine(fiducialPointMatrix , resultAffineMatrix, affineMatrix, resultAffineMatrix.size()); // not working
            mean[landmarkIterator].x += (double(resultAffineMatrix.at<double>(0,0)))/landmarks.size();
            mean[landmarkIterator].y += (double(resultAffineMatrix.at<double>(1,0)))/landmarks.size();
            landmarkIterator++;
        }
    }
    //fill meanShape vector
    ofstream meanShapefile;
    meanShapefile.open("meanshape.txt");
    meanShapefile << "%% MeanShape Landmark Locations\n" ;
    for (int fillMeanShape = 0; fillMeanShape < numLandmarks; ++fillMeanShape)
    {
        meanShape.push_back(mean[fillMeanShape]);
        meanShapefile << mean[fillMeanShape].x << ","<< mean[fillMeanShape].y << "\n";

    }
    return true;
}

bool KazemiFaceAlignImpl::getInitialShape(Mat& image, CascadeClassifier& cascade)
{
    if(image.empty() || meanShape.empty())
    {
        String error_message = "The data is not loaded properly by train function. Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    //find bounding rectangle for the mean shape
    // Mat meanX(meanShape.size(),1,CV_8UC1);
    // Mat meanY(meanShape.size(),1,CV_8UC1);
    // int pointcount = 0 ;
    // for (vector<Point2f>::iterator it = meanShape.begin(); it != meanShape.end(); ++it)
    // {
    //     meanX.at<double>(pointcount,0) = (*it).x;
    //     meanY.at<double>(pointcount,0) = (*it).y;
    //     pointcount++;
    // }
    // //find max and min x and y
    double meanShapeRectminx, meanShapeRectminy, meanShapeRectmaxx, meanShapeRectmaxy;
    // minMaxLoc(meanX, &meanShapeRectminx, &meanShapeRectmaxx);
    // minMaxLoc(meanY, &meanShapeRectminy, &meanShapeRectmaxy);
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
    Point2f refrencePoints[3];
    refrencePoints[0] = Point2f(meanShapeRectminx, meanShapeRectminy);
    refrencePoints[1] = Point2f(meanShapeRectmaxx, meanShapeRectminy);
    refrencePoints[2] = Point2f(meanShapeRectminx, meanShapeRectmaxy);
    //apply face detector on the current image
    vector<Rect> facesInImage = faceDetector(image , cascade);
    for(unsigned int facenum =0 ; facenum < facesInImage.size(); facenum++)
    {
        Point2f currentPoints[3];
        currentPoints[0] = Point2f(facesInImage[facenum].x,facesInImage[facenum].y);
        currentPoints[1] = Point2f((facesInImage[facenum].x + facesInImage[facenum].width),facesInImage[facenum].y);
        currentPoints[2] = Point2f(facesInImage[facenum].x , (facesInImage[facenum].y + facesInImage[facenum].height));
        Mat affineMatrix = getAffineTransform(refrencePoints , currentPoints);
        vector<Point2f> intermediate;
        //Transform each fiducial Point to get it relative to the initital image
        for (vector< Point2f >::iterator fiducialIt = meanShape.begin() ; fiducialIt != meanShape.end() ; fiducialIt++ )
        {
            Mat fiducialPointMatrix = (Mat_<double>(3,1) << (*fiducialIt).x, (*fiducialIt).y , 1);
            Mat resultAffineMatrix = (Mat_<double>(3,1)<<0,0,1);
            resultAffineMatrix = affineMatrix*fiducialPointMatrix;
            //warpAffine(fiducialPointMatrix , resultAffineMatrix, affineMatrix, resultAffineMatrix.size()); // not working
            intermediate.push_back(Point2f(resultAffineMatrix.at<double>(0,0) , resultAffineMatrix.at<double>(1,0)));
        }
        initialShape.push_back(intermediate);
    }
return true;
}



bool KazemiFaceAlignImpl::extractPixelValues(trainSample& sample , vector<Point2f>& pixelCoordinates)
{
    Mat image = sample.img;
    sample.pixelValues.resize(pixelCoordinates.size());
    if(image.channels() != 1)
        cvtColor(image,image,COLOR_BGR2GRAY);
    for (unsigned int i = 0; i < pixelCoordinates.size(); ++i)
    {
        if(pixelCoordinates[i].x < image.rows && pixelCoordinates[i].y < image.cols)
        {
            sample.pixelValues[i] = (int)image.at<uchar>(pixelCoordinates[i].x, pixelCoordinates[i].y);
        }
    }
    return true;
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

bool KazemiFaceAlignImpl::calcMul(vector<Point2f>& input1, vector<Point2f>& input2, vector<Point2f>& output)
{
    output.resize(input1.size());
    for (unsigned long i = 0; i < input1.size(); ++i)
    {
        output[i].x = input1[i].x * input2[i].x;
        output[i].y = input1[i].y * input2[i].y;
    }
    return true;;
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

bool KazemiFaceAlignImpl::getRelativeShapefromMean(trainSample& sample, vector<Point2f>& landmarks)
{
    for(unsigned int facenum =0 ; facenum < sample.rect.size(); facenum++)
    {
        Point2f currentPoints[3];
        currentPoints[0] = Point2f(sample.rect[facenum].x,sample.rect[facenum].y);
        currentPoints[1] = Point2f((sample.rect[facenum].x + sample.rect[facenum].width),sample.rect[facenum].y);
        currentPoints[2] = Point2f(sample.rect[facenum].x , (sample.rect[facenum].y + sample.rect[facenum].height));
        Mat affineMatrix = getAffineTransform( meanShapeReferencePoints, currentPoints);
        vector<Point2f> intermediate;
        int counter = 0;
        //Transform each fiducial Point to get it relative to the initital image
        for (vector< Point2f >::iterator fiducialIt = landmarks.begin() ; fiducialIt != landmarks.end() ; fiducialIt++ )
        {
            Mat fiducialPointMatrix = (Mat_<double>(3,1) << (*fiducialIt).x, (*fiducialIt).y , 1);
            Mat resultAffineMatrix = (Mat_<double>(3,1)<<0,0,1);
            resultAffineMatrix = affineMatrix*fiducialPointMatrix;
            sample.currentShape.push_back(Point2f(resultAffineMatrix.at<double>(0,0) , resultAffineMatrix.at<double>(1,0)));
        }
    }
    return true;
}


bool KazemiFaceAlignImpl::getRelativeShapetoMean(trainSample& sample, vector<Point2f>& landmarks)
{
    for(unsigned int facenum =0 ; facenum < sample.rect.size(); facenum++)
    {
        Point2f currentPoints[3];
        currentPoints[0] = Point2f(sample.rect[facenum].x,sample.rect[facenum].y);
        currentPoints[1] = Point2f((sample.rect[facenum].x + sample.rect[facenum].width),sample.rect[facenum].y);
        currentPoints[2] = Point2f(sample.rect[facenum].x , (sample.rect[facenum].y + sample.rect[facenum].height));
        Mat affineMatrix = getAffineTransform(  currentPoints, meanShapeReferencePoints);
        vector<Point2f> intermediate;
        int counter = 0;
        //Transform each fiducial Point to get it relative to the initital image
        for (vector< Point2f >::iterator fiducialIt = landmarks.begin() ; fiducialIt != landmarks.end() ; fiducialIt++ )
        {
            Mat fiducialPointMatrix = (Mat_<double>(3,1) << (*fiducialIt).x, (*fiducialIt).y , 1);
            Mat resultAffineMatrix = (Mat_<double>(3,1)<<0,0,1);
            resultAffineMatrix = affineMatrix*fiducialPointMatrix;
            sample.currentShape.push_back(Point2f(resultAffineMatrix.at<double>(0,0) , resultAffineMatrix.at<double>(1,0)));
        }
    }
    return true;
}

}
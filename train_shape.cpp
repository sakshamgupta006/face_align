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
#include "../include/opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "train_shape.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

namespace cv{

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

unsigned long leftChild(unsigned long idx)
{
    return 2*idx + 1;
}

unsigned long rightChild(unsigned long idx)
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
bool KazemiFaceAlignImpl::readtxt(vector<cv::String>& filepath, std::map<string, vector<Point2f>>& landmarks, string path_prefix)
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
    return true;
}

vector<Rect> KazemiFaceAlignImpl::faceDetector(Mat image,CascadeClassifier& cascade)
{
    vector<Rect> faces;
    int scale = 1;
    //remove once testing is complete
    Mat gray, smallImg;
    cvtColor( image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray,gray);
    cascade.detectMultiScale( gray, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT,
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    numFaces = faces.size();
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Scalar color = Scalar(255,0,0);
        double aspect_ratio = (double)r.width/r.height;
        rectangle( image, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
    }
    return faces;
}

Mat KazemiFaceAlignImpl::getImage(string imgpath, string path_prefix)
{
    return imread(path_prefix + imgpath + ".jpg");
}

bool KazemiFaceAlignImpl::extractMeanShape(std::map<string, vector<Point2f>>& landmarks, string path_prefix,CascadeClassifier& cascade)
{
    //string meanShapeFileName = "mean_shape.xml";
    std::map<string, vector<Point2f>>::iterator dbIterator = landmarks.begin(); // random inititalization as
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
        cout<<affineMatrix<<endl;
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

double KazemiFaceAlignImpl::getDistance(Point2f first , Point2f second)
{
    return sqrt(pow((first.x - second.x),2) + pow((first.y - second.y),2));
}

splitFeature KazemiFaceAlignImpl::randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates)
{
    splitFeature feature;
    double acceptProbability;
    RNG rnd;
    do
    {
        feature.idx1 = rnd.uniform(0,numFeature);
        feature.idx2 = rnd.uniform(0,numFeature);
        double dist = getDistance(pixelCoordinates[feature.idx1] , pixelCoordinates[feature.idx2]);
        acceptProbability = exp(-dist/lambda);
    }
    while(feature.idx1 == feature.idx2 || !(acceptProbability > rnd.uniform(0,1)));
    feature.thresh = rnd.uniform(double(-255),double(255)); //Check Validity
    return feature;
}

splitFeature KazemiFaceAlignImpl::splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long start ,
                                                unsigned long end,const Point2f& sum, Point2f& leftSum, Point2f& rightSum)
{
    vector<splitFeature> features;
    for (unsigned int i = 0; i < numTestSplits; ++i)
        features.push_back(randomSplitFeatureGenerator(pixelCoordinates));
    vector<Point2f> leftSums;
    vector<unsigned long> leftCount;
    ///////--------------------SCOPE OF THREADING----------------------/////////////
    for (unsigned long i = start; i < end ; ++i)
    {
        for (unsigned long j = 0; j < numTestSplits; ++j)
        {
            if((float)samples[i].pixelValues[features[j].idx1] - (float)samples[i].pixelValues[features[j].idx2] > features[j].thresh)
            {
                leftSums[j] = calcSum(leftSums[j], samples[i].residualShape);
                ++leftCount[j];
            }
        }
    }
    ///////--------------------SCOPE OF THREADING----------------------/////////////
    //Select the best feature
    double bestScore = -1;
    unsigned long bestFeature = 0;
    Point2f tempFeature;
    for (unsigned long i = 0; i < numTestSplits; ++i)
    {
        double currentScore = 0;
        unsigned long rightCount = end - start - leftCount[i];
        if(leftCount[i] != 0 && rightCount != 0)
        {
            tempFeature = sum - leftSums[i];
            //To calculate score
            double leftSumsDot = pow(leftSums[i].x,2) + pow(leftSums[i].y,2);
            double tempFeatureDot = pow(tempFeature.x,2) + pow(tempFeature.y,2);
            currentScore = leftSumsDot/leftCount[i] + tempFeatureDot/rightCount;
            if(currentScore > bestScore)
            {
                bestScore = currentScore;
                bestFeature = i;
            }
        }
    }
    //Swap the Coordinate Values
    Point2f temp = leftSums[bestFeature];
    leftSums[bestFeature] = leftSum;
    leftSum = temp;
    //leftSums[bestFeature].swap(leftSum);
    if(leftSum.x != 0 && leftSum.y !=0)
        rightSum = sum - leftSum;
    else
    {
        rightSum = sum;
        leftSum = Point2f(0,0);
    }
return features[bestFeature];
}

bool KazemiFaceAlignImpl::extractPixelValues(trainSample &sample , vector<Point2f> pixelCoordinates)
{
    Mat image = sample.img;
    for (unsigned int i = 0; i < pixelCoordinates.size(); ++i)
    {
        cvtColor(image,image,COLOR_BGR2GRAY);
        sample.pixelValues.push_back(image.at<double>(pixelCoordinates[i]));
    }
    return true;
}

regressionTree KazemiFaceAlignImpl::buildRegressionTree(vector<trainSample>& samples, vector<Point2f> pixelCoordinates)
{
    regressionTree tree;
    //partition queue will store the extent of leaf nodes
    deque< pair<unsigned long, unsigned long > >  partition;
    partition.push_back(make_pair(0, (unsigned long)samples.size()));
    const unsigned long numSplitNodes = (unsigned long)(pow(2 , (double)getTreeDepth()) - 1);
    const unsigned long numLeaves =
    vector<Point2f> sums(numSplitNodes*2 + 1);
    ////---------------------------------SCOPE OF THREADING---------------------------------------////
    for (unsigned long i = 0; i < samples.size(); i++)
    {
        samples[i].residualShape = calcDiff(samples[i].targetShape , samples[i].currentShape);
        sums = calcSum(sums,samples[i].residualShape);
    }
    //Iteratively generate Splits in the samples
    for (unsigned long int i = 0; i < numSplitNodes; i++)
    {
        pair<unsigned long, unsigned long> rangeleaf = partition.front();
        partition.pop_front();
        splitFeature split = splitGenerator(samples, pixelCoordinates, rangeleaf.first, rangeleaf.second, sums[i], sums[leftChild(i)], sums[rightChild(i)]);
        tree.split.push_back(split);
        const unsigned long mid = partitionSamples(split, samples, rangeleaf.first, rangeleaf.second);
        partition.push_back(make_pair(rangeleaf.first, mid));
        partition.push_back(make_pair(mid, rangeleaf.second));
    }
    tree.leaves.resize(partition.size());
    //Use partition value to calculate average value of leafs
    for (unsigned long int i = 0; i < partition.size(); ++i)
    {
        unsigned long currentCount = partition[i].second - partition[i].first + 1;
        vector<Point2f> residualSum;
        for (unsigned long j = partition[i].first; j < partition[i].second; ++j)
            residualSum = calcSum(residualSum, samples[j].residualShape);
        for (unsigned long k = 0; k < residualSum.size(); ++k)
        {
            if(partition[i].first != partition[i].second)
            {
                residualSum[j].x /= currentCount;
                residualSum[j].y /= currentCount;
            }
        }
        tree.leaves[i] = residualSum;
        for (unsigned long j = partition[i].first; j < partition[i].second; ++j)
        {
            for (unsigned long k = 0; k < samples[j].residualShape; ++k)
            {
                samples[j].residualShape[k] -= learningRate * tree.leaves[i][k];
            }
        }
    }
    return tree;
}

unsigned long KazemiFaceAlignImpl::partitionSamples(splitFeature split, vector<trainSample>& samples, unsigned long start, unsigned long end)
{
    unsigned long initial = start;
    for (unsigned long j = 0; j < end; j++)
    {
        if((float)samples[j].pixelValues[split.idx1] - (float)samples[j].pixelValues[split.idx2] > split.thresh)
        {
            swap(samples[initial],samples[j]);
            initial++;
        }
    }
    return initial;
}

vector<Point2f> calcDiff(vector<Point2f> target, vector<Point2f> current)
{
    vector<Point2f> resultant;
    for (unsigned long i = 0; i < target.size(); ++i)
    {
        resultant[i] = target[i] - current[i];
    }
    return resultant;
}

vector<Point2f> calcSum(vector<Point2f> target, vector<Point2f> current)
{
    vector<Point2f> resultant;
    for (unsigned long i = 0; i < target.size(); ++i)
    {
        resultant[i] = target[i] + current[i];
    }
    return resultant;
}

vector<regressionTree> KazemiFaceAlignImpl::gradientBoosting(vector<trainSample>& samples, vector<Point2f> pixelCoordinates)
{
    //for cascade of regressrs
    vector<regressionTree> forest;
    for (unsigned long i = 0; i < numTreesperCascade; ++i)
    {
        regressionTree tree = buildRegressionTree(samples,pixelCoordinates);
        forest.push_back(tree);
    }
    for (unsigned long i = 0; i < samples.size(); i++)
    {
       samples[i].currentShape = calcDiff(samples[i].targetShape, samples[i].residualShape);
    }
    return forest;
}

bool KazemiFaceAlignImpl::trainCascade()
{
    vector<trainingSample> samples;
    vector< vector<Point2f> > pixelCoordinates;
    fillData(samples,pixelCoordinates);
    return true;
}

}
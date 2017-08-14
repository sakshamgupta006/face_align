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

#define numSamples 100

namespace cv
{


class relativeSample : public ParallelLoopBody, protected KazemiFaceAlignImpl
{
public:
    relativeSample (vector<trainSample>& samples, vector<Point2f> pixrel, vector<Point2f>& meanShape)
        : _samples(samples), _pixrel(pixrel), _meanShape(meanShape)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        KazemiFaceAlignImpl calculations;
        for (unsigned long r = range.start; r < range.end; r++)
        {
            _samples[r].pixelCoordinates = _pixrel;
            calculations.calcRelativePixelsParallel(_samples[r], _meanShape);
            calculations.extractPixelValues(_samples[r]);
        }
    }

private:
    vector<trainSample>& _samples;
    vector<Point2f> _pixrel;
    vector<Point2f>& _meanShape;
};

bool KazemiFaceAlignImpl::calcRelativePixelsParallel(trainSample& sample, vector<Point2f>& meanShape)
{
    if(sample.currentShape.size()!= meanShape.size())
    {
        String errmsg = "Shape Size Mismatch Detected";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    Mat affineMatrix = estimateRigidTransform(sample.currentShape, meanShape, false);
    if(affineMatrix.empty())
    {
        double sampleShapeRectminx, sampleShapeRectminy, sampleShapeRectmaxx, sampleShapeRectmaxy;
        double sampleX[sample.currentShape.size()] , sampleY[sample.currentShape.size()];
        int pointcount = 0;
        for (vector<Point2f>::iterator it = sample.currentShape.begin(); it != sample.currentShape.end(); ++it)
        {
            sampleX[pointcount] = (*it).x;
            sampleY[pointcount] = (*it).y;
            pointcount++;
        }
        sampleShapeRectminx = *min_element(sampleX , sampleX + sample.currentShape.size());
        sampleShapeRectmaxx = *max_element(sampleX , sampleX + sample.currentShape.size());
        sampleShapeRectminy = *min_element(sampleY , sampleY + sample.currentShape.size());
        sampleShapeRectmaxy = *max_element(sampleY , sampleY + sample.currentShape.size());
        Point2f sampleRefPoints[3];
        sampleRefPoints[0] = Point2f(sampleShapeRectminx , sampleShapeRectminy );
        sampleRefPoints[1] = Point2f( sampleShapeRectmaxx, sampleShapeRectminy );
        sampleRefPoints[2] = Point2f( sampleShapeRectminx, sampleShapeRectmaxy );
        affineMatrix = getAffineTransform(sampleRefPoints, meanShapeReferencePoints);
    }
    for(unsigned long i = 0; i < sample.pixelCoordinates.size(); i++)
    {
        unsigned long in = findNearestLandmark(sample.pixelCoordinates[i]);
        Point2f point = sample.pixelCoordinates[i] - meanShape[in];
        Mat fiducialPointMat = (Mat_<double>(3,1) << point.x, point.y, 0);
        Mat resultAffineMat = affineMatrix * fiducialPointMat;
        point.x = float((resultAffineMat.at<double>(0,0)));
        point.y = float((resultAffineMat.at<double>(1,0)));
        sample.pixelCoordinates[i] = point + sample.pixelCoordinates[in];
    }
    return true;
}


Mat KazemiFaceAlignImpl::normalizing_tform(Rect& r)
{
    Point2f from_points[3], to_points[3];
    from_points[1] = Point2f(r.x + r.width, r.y);
    from_points[0] = Point2f(r.x, r.y);
    from_points[2] = Point2f(r.x + r.width, r.y + r.height);
    to_points[0] = Point2f(0,0);
    to_points[1] = Point2f(1,0);
    to_points[2] = Point2f(1,1);
    return getAffineTransform(from_points, to_points);
}

Mat KazemiFaceAlignImpl::unnormalizing_tform(Rect& r)
{
    Point2f from_points[3], to_points[3];
    from_points[0] = Point2f(0,0);
    from_points[1] = Point2f(1,0);
    from_points[2] = Point2f(1,1);
    to_points[0] = Point2f(r.x, r.y);
    to_points[1] = Point2f(r.x + r.width, r.y);
    to_points[2] = Point2f(r.x + r.width, r.y + r.height);
    return getAffineTransform(from_points, to_points);
}

void KazemiFaceAlignImpl::savesample(trainSample samples, int no)
{
 Mat image = samples.img.clone();
    Mat unorm_tform  = unnormalizing_tform(samples.rect[0]);
    vector<Point2f> some(numLandmarks, Point2f(0.,0.));
        for (int j = 0; j < samples.currentShape.size(); ++j)
        {
            Mat temp = (Mat_<double>(3,1)<< samples.currentShape[j].x , samples.currentShape[j].y , 1);
            Mat res = unorm_tform * temp;
            some[j].x = res.at<double>(0,0);
            some[j].y = res.at<double>(1,0);
        }
    for (int j = 0; j < samples.targetShape.size() ; ++j)
    {
        circle(image, Point(some[j]), 2, Scalar(255,0,0), -1 );
    }   
    string saves = "res" + to_string(no) + ".png";
    imwrite(saves, image);
}

bool KazemiFaceAlignImpl::extractPixelValues(trainSample& sample)
{
    vector<Point2f> pixels(sample.pixelCoordinates.size());
    Mat unormmat = unnormalizing_tform(sample.rect[0]);
    for(unsigned long i = 0; i < sample.pixelCoordinates.size(); i++)
    {
        Mat fiducialPointMat = (Mat_<double>(3,1) << sample.pixelCoordinates[i].x, sample.pixelCoordinates[i].y, 1);
        Mat resultAffineMat = unormmat * fiducialPointMat;
        pixels[i].x = float((resultAffineMat.at<double>(0,0)));
        pixels[i].y = float((resultAffineMat.at<double>(1,0)));
    }
    Mat image = sample.img.clone();
    sample.pixelValues.resize(sample.pixelCoordinates.size());
    if(image.channels() != 1)
        cvtColor(image,image,COLOR_BGR2GRAY);
    for (unsigned int i = 0; i < sample.pixelCoordinates.size(); i++)
    {
        if(pixels[i].x >= 0 && pixels[i].x < image.cols && pixels[i].y >= 0 && pixels[i].y < image.rows)
            sample.pixelValues[i] = image.at<uchar>(pixels[i].y, pixels[i].x);
        else
            sample.pixelValues[i] = 0;
    }
    return true;
}

bool KazemiFaceAlignImpl::trainCascade(std::unordered_map<string, vector<Point2f>>& landmarks, string path_prefix, CascadeClassifier& cascade, string outputName)
{
    double total_time = 0, t = 0;
    vector<trainSample> samples;
    vector< vector<Point2f> > pixelCoordinates;
    fillData(samples, landmarks, path_prefix, cascade);
    generateTestCoordinates(pixelCoordinates);
    ofstream fs(outputName, ios::out | ios::binary);
    if (!fs.is_open())
    {
        cerr << "Cannot open binary file to save the model"<< endl;
        return false;
    }
    vector< vector<regressionTree> > cascadeFinal;
    cout<<"Training Started"<<endl;
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        t = (double)getTickCount();
        //parallel_for_(Range(0, samples.size()), relativeSample(samples, pixelCoordinates[i], meanShape));
        for (unsigned long j = 0; j < samples.size(); ++j)
        {
            samples[j].pixelCoordinates = pixelCoordinates[i];
            calcRelativePixels(samples[j]);
            extractPixelValues(samples[j]);
        }
        vector<regressionTree> forest = gradientBoosting(samples, pixelCoordinates[i]);
        cascadeFinal.push_back(forest);
        cout<<"Fitted "<< i + 1 <<"th regressor"<<endl;
        t = (double)getTickCount() - t;
        total_time += t;
        cout<<"Time Taken to fit Cascade = "<< t/(getTickFrequency()*60) <<" min"<<endl;
    }
    cout<<"Total training time = "<< total_time/(getTickFrequency()*60*60) <<" hrs"<<endl;
    writeModel(fs,cascadeFinal, pixelCoordinates);
    fs.close();
    for (int i = 0; i < samples.size(); ++i)
    {
        renderDetections(samples[i], Scalar(0,255,0), 2);
    }
    return true;
}

bool KazemiFaceAlignImpl::fillData(vector<trainSample>& samples,std::unordered_map<string, vector<Point2f>>& landmarks,
                                    string path_prefix, CascadeClassifier& cascade)
{   cout<<"Inside filldata"<<endl;
    meanShape.assign(numLandmarks, Point2f(0.,0.));
    unsigned long currentCount = 0;
    for (unordered_map<string, vector<Point2f>>::iterator dbIterator = landmarks.begin();
            dbIterator != landmarks.end(); ++dbIterator)
    {   
        if(currentCount > numSamples)
            break;
        trainSample sample;
        sample.img =  imread(dbIterator->first);//getImage(dbIterator->first,path_prefix);
        cout<<dbIterator->first<<endl;
        sample.targetShape.resize(numLandmarks);
        sample.targetShape = dbIterator->second;
        scaleData(sample.targetShape, sample.img,  Size(460,460));
        sample.rect = faceDetector(sample.img, cascade);
        if(sample.rect.size() != 1)
        {
            continue;
        }
        Mat normMat = normalizing_tform(sample.rect[0]);
        for (unsigned long j = 0; j < sample.targetShape.size(); ++j)
        {
            Mat targetshapepoint = (Mat_<double>(3,1) << (sample.targetShape[j].x) , (sample.targetShape[j].y) , 1);
            Mat multargetshapepoint = normMat * targetshapepoint;
            sample.targetShape[j].x = multargetshapepoint.at<double>(0,0);
            sample.targetShape[j].y = multargetshapepoint.at<double>(1,0);
        }
        for (unsigned long j = 0; j < oversamplingAmount; ++j)
            samples.push_back(sample);
        calcSum(sample.targetShape, meanShape, meanShape);
        currentCount++;
    }
    for (int i = 0; i < meanShape.size(); ++i)
    {
        meanShape[i].x /= currentCount;
        meanShape[i].y /= currentCount;
    }
    calcMeanShapeBounds();
    for (unsigned long i = 0; i < samples.size(); ++i)
    {
        samples[i].currentShape.assign(meanShape.size(), Point2f(0.,0.));
        samples[i].residualShape.resize(meanShape.size(), Point2f(0.,0.));
        if(i%oversamplingAmount == 0)
            samples[i].currentShape = meanShape;
        else
        {
            double hits = 0;
            int count = 0;
            for (int randomint = 0; randomint < 4; ++randomint)
            {
                    RNG number(getTickCount());
                    unsigned long randomIndex = (unsigned long)number.uniform(0, currentCount*oversamplingAmount-1);
                    for (unsigned long j = 0; j < meanShape.size(); ++j)
                    {
                        samples[i].currentShape[j].x += samples[randomIndex].targetShape[j].x;
                        samples[i].currentShape[j].y += samples[randomIndex].targetShape[j].y;
                    }
                    count++;
            }
            for (unsigned long l = 0; l < samples[i].currentShape.size(); ++l)
            {
                samples[i].currentShape[l].x /= count;
                samples[i].currentShape[l].y /= count;
            }
        }
    }
    cout<<"Total Images Loaded -> "<<(currentCount-1) <<endl;
    cout<<"Total Sample Size -> "<< samples.size() <<endl;
    return true;
}

bool KazemiFaceAlignImpl::scaleData(vector<Point2f>& landmarks, Mat& image, Size s)
{
    float scalex,scaley;
    scalex = (float)s.width/image.cols;
    scaley = (float)s.height/image.rows;
    resize(image, image, s);
    for (unsigned long i = 0; i < landmarks.size(); i++)
    {
        landmarks[i].x *= scalex;
        landmarks[i].y *= scaley;
    }
    return true;
}

bool KazemiFaceAlignImpl::displayresultstarget(trainSample& samples)
{
        vector<Point2f> temp1(samples.targetShape.size());
        Mat image = samples.img.clone();
        Mat unorm_tform  = unnormalizing_tform(samples.rect[0]);
        for (int j = 0; j < samples.targetShape.size(); ++j)
        {
            Mat temp = (Mat_<double>(3,1)<< samples.targetShape[j].x , samples.targetShape[j].y , 1);
            Mat res = unorm_tform * temp;
            temp1[j].x = res.at<double>(0,0);
            temp1[j].y = res.at<double>(1,0);
        }

        for (int j = 0; j < samples.targetShape.size() ; ++j)
        {
            circle(image, Point(temp1[j]), 5, Scalar(0,0,255) ,-1);
        }
        imshow("Results", image);
        waitKey(0);
    return true;
}

bool KazemiFaceAlignImpl::displayresultstarget2(vector<trainSample>& samples)
{
    for (int i = 0; i < samples.size(); ++i)
     {
        vector<Point2f> temp1(samples[i].targetShape.size());
        Mat image = samples[i].img.clone();
        Mat unorm_tform  = unnormalizing_tform(samples[i].rect[0]);
        for (int j = 0; j < samples[i].targetShape.size(); ++j)
        {
            Mat temp = (Mat_<double>(3,1)<< samples[i].targetShape[j].x , samples[i].targetShape[j].y , 1);
            Mat res = unorm_tform * temp;
            temp1[j].x = res.at<double>(0,0);
            temp1[j].y = res.at<double>(1,0);
        }

        for (int j = 0; j < samples[i].targetShape.size() ; ++j)
        {
            circle(image, Point(temp1[j]), 5, Scalar(0,0,255) ,-1);
        }
        imshow("Results", image);
        waitKey(0);
    }
    return true;
}


bool KazemiFaceAlignImpl::displayresults2(vector<trainSample>& samples)
{
    for (int i = 0; i < samples.size(); ++i)
     {
        vector<Point2f> temp1(samples[i].targetShape.size());
        Mat image = samples[i].img.clone();
        Mat unorm_tform  = unnormalizing_tform(samples[i].rect[0]);
        for (int j = 0; j < samples[i].targetShape.size(); ++j)
        {
            Mat temp = (Mat_<double>(3,1)<< samples[i].currentShape[j].x , samples[i].currentShape[j].y , 1);
            Mat res = unorm_tform * temp;
            temp1[j].x = res.at<double>(0,0);
            temp1[j].y = res.at<double>(1,0);
        }

        for (int j = 0; j < samples[i].targetShape.size() ; ++j)
        {
            circle(image, Point(temp1[j]), 5, Scalar(0,0,255) ,-1);
        }
        imshow("Display Results 2 ", image);
        waitKey(0);
    }
    return true;
}

bool KazemiFaceAlignImpl::displayresults(trainSample& samples)
{
    Mat image = samples.img.clone();
    Mat unorm_tform  = unnormalizing_tform(samples.rect[0]);
    vector<Point2f> temp1;
    temp1.resize(samples.currentShape.size());
        for (int j = 0; j < samples.currentShape.size(); ++j)
        {
            Mat temp = (Mat_<double>(3,1)<< samples.currentShape[j].x , samples.currentShape[j].y , 1);
            Mat res = unorm_tform * temp;
            temp1[j].x = (float)(res.at<double>(0,0));
            temp1[j].y = (float)(res.at<double>(1,0));
        }
    for (int j = 0; j < samples.currentShape.size() ; ++j)
    {
        circle(image, Point(temp1[j]), 3, Scalar(0,0,255), -1 );
    }
    imshow("Display Results", image);
    //waitKey(0);
    //imwrite("result.jpg", image);
    return true;
}

bool KazemiFaceAlignImpl::generateTestCoordinates(vector< vector<Point2f> >& pixelCoordinates)
{
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        RNG rng(time(0));
        vector<Point2f> testCoordinates;
        for (unsigned long j = 0; j < numTestCoordinates; ++j)
        {
            testCoordinates.push_back(Point2f((float)rng.uniform(meanShapeBounds[0].x, meanShapeBounds[1].x), (float)rng.uniform(meanShapeBounds[0].y,meanShapeBounds[1].y)));
        }
        pixelCoordinates.push_back(testCoordinates);
    }
    return true;
}

unsigned int KazemiFaceAlignImpl::findNearestLandmark(Point2f& pixelValue)
{
    float minDist = INT_MAX;
    unsigned int value =0;
    for(unsigned int it = 0; it < meanShape.size(); it++)
    {
        Point2f currentPoint = pixelValue - meanShape[it];
        float calculatedDiff = sqrt(pow(currentPoint.x,2) + pow(currentPoint.y,2));
        if(calculatedDiff < minDist)
            {
                minDist = calculatedDiff;
                value = it;
            }
    }
    return value;
}

bool KazemiFaceAlignImpl::calcRelativePixels(trainSample& sample)
{
    if(sample.currentShape.size()!= meanShape.size())
    {
        String errmsg = "Shape Size Mismatch Detected";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    Mat affineMatrix = estimateRigidTransform(sample.currentShape, meanShape, false);
    if(affineMatrix.empty())
    {
        double sampleShapeRectminx, sampleShapeRectminy, sampleShapeRectmaxx, sampleShapeRectmaxy;
        double sampleX[sample.currentShape.size()] , sampleY[sample.currentShape.size()];
        int pointcount = 0;
        for (vector<Point2f>::iterator it = sample.currentShape.begin(); it != sample.currentShape.end(); ++it)
        {
            sampleX[pointcount] = (*it).x;
            sampleY[pointcount] = (*it).y;
            pointcount++;
        }
        sampleShapeRectminx = *min_element(sampleX , sampleX + sample.currentShape.size());
        sampleShapeRectmaxx = *max_element(sampleX , sampleX + sample.currentShape.size());
        sampleShapeRectminy = *min_element(sampleY , sampleY + sample.currentShape.size());
        sampleShapeRectmaxy = *max_element(sampleY , sampleY + sample.currentShape.size());
        Point2f sampleRefPoints[3];
        sampleRefPoints[0] = Point2f(sampleShapeRectminx , sampleShapeRectminy );
        sampleRefPoints[1] = Point2f( sampleShapeRectmaxx, sampleShapeRectminy );
        sampleRefPoints[2] = Point2f( sampleShapeRectminx, sampleShapeRectmaxy );
        affineMatrix = getAffineTransform(sampleRefPoints, meanShapeReferencePoints);
    }
    for(unsigned long i = 0; i < sample.pixelCoordinates.size(); i++)
    {
        unsigned long in = findNearestLandmark(sample.pixelCoordinates[i]);
        Point2f point = sample.pixelCoordinates[i] - meanShape[in];
        Mat fiducialPointMat = (Mat_<double>(3,1) << point.x, point.y, 0);
        Mat resultAffineMat = affineMatrix * fiducialPointMat;
        point.x = float((resultAffineMat.at<double>(0,0)));
        point.y = float((resultAffineMat.at<double>(1,0)));
        sample.pixelCoordinates[i] = point + sample.pixelCoordinates[in];
    }
    return true;
}

void KazemiFaceAlignImpl::writeSplit(ofstream& fs, vector<splitFeature>& split)
{
    unsigned long splitsize = split.size();
    fs.write(reinterpret_cast<const char *>(&splitsize), sizeof(unsigned long));
    for(unsigned long i = 0; i<split.size();i++)
    {
        string splitFeatureString = "Split_Feature";
        size_t lenSplitFeatureString = splitFeatureString.size();
        fs.write((char*)&lenSplitFeatureString, sizeof(size_t));
        fs.write(splitFeatureString.c_str(), lenSplitFeatureString);
        fs.write((char*)&split[i], sizeof(splitFeature));
    }
}

void KazemiFaceAlignImpl::writeLeaf(ofstream& fs, vector< vector<Point2f> >& leaves)
{
    unsigned long leavessize = leaves.size();
    fs.write(reinterpret_cast<const char *>(&leavessize), sizeof(unsigned long));
    for (unsigned long j = 0; j < leaves.size(); ++j)
    {
        string leafString = "Leaf";
        size_t lenleafString = leafString.size();
        fs.write((char*)&lenleafString, sizeof(size_t));
        fs.write(leafString.c_str(), lenleafString);
        size_t leafSize = leaves[j].size();
        fs.write((char*)&leafSize, sizeof(leafSize));
        fs.write((char*)&leaves[j][0], sizeof(Point2f)*leaves[j].size());
    }
}

void KazemiFaceAlignImpl::writeTree(ofstream& fs, regressionTree& tree)
{
    writeSplit(fs,tree.split);
    writeLeaf(fs,tree.leaves);
}

void KazemiFaceAlignImpl::writeCascade(ofstream& fs, vector<regressionTree>& forest)
{
    for (unsigned long j = 0; j < forest.size(); ++j)
    {
        writeTree(fs,forest[j]);
    }
}

void KazemiFaceAlignImpl::writeModel(ofstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> > pixelCoordinates)
{
    //Writing MeanShape START
    string meanShapestring = "Mean_Shape";
    size_t lenMeanShapeString = meanShapestring.size();
    fs.write((char*)&lenMeanShapeString, sizeof(size_t));
    fs.write(meanShapestring.c_str(), lenMeanShapeString);
    size_t meanShapeSize = meanShape.size();
    fs.write(reinterpret_cast<const char *>(&meanShapeSize), sizeof(meanShapeSize));
    fs.write((char*)&meanShape[0], sizeof(Point2f)*meanShape.size());
    //MeanShape Writing END

    //Write Pixel Coordinates START
    string pixelCoordinatesString = "Pixel_Coordinates";
    size_t lenpixelCoordinatesString = pixelCoordinatesString.size();
    fs.write((char*)&lenpixelCoordinatesString, sizeof(size_t));
    fs.write(pixelCoordinatesString.c_str(), lenpixelCoordinatesString);
    size_t pixelCoordinateMainVecSize = pixelCoordinates.size();
    fs.write(reinterpret_cast<const char *>(&pixelCoordinateMainVecSize), sizeof(pixelCoordinateMainVecSize));
    for (unsigned long i = 0; i < pixelCoordinates.size(); ++i)
    {
        size_t pixelCoordinateVecSize = pixelCoordinates[i].size();
        fs.write(reinterpret_cast<const char *>(&pixelCoordinateVecSize), sizeof(pixelCoordinateVecSize));
        fs.write((char*)&pixelCoordinates[i][0], sizeof(Point2f)*pixelCoordinates[i].size());
    }
    //Write Pixel Coordinates END

    //Write Cascade
    string cascadeDepthString = "Cascade_Depth";
    size_t lenCascadeDepthString = cascadeDepthString.size();
    fs.write((char*)&lenCascadeDepthString, sizeof(size_t));
    fs.write(cascadeDepthString.c_str(), lenCascadeDepthString);
    fs.write(reinterpret_cast<const char *>(&cascadeDepth), sizeof(cascadeDepth));
    //Number of Trees in each cascade
    string numTreesperCascadeString = "Num_Trees_per_Cascade";
    size_t lennumTreesperCascadeString = numTreesperCascadeString.size();
    fs.write((char*)&lennumTreesperCascadeString, sizeof(size_t));
    fs.write(numTreesperCascadeString.c_str(), lennumTreesperCascadeString);
    fs.write(reinterpret_cast<const char *>(&numTreesperCascade), sizeof(numTreesperCascade));
    //Now write each Tree
    for (unsigned long i = 0; i < forest.size(); ++i)
    {
        writeCascade(fs, forest[i]);
    }
}

/*+++++++++++++++++++++++++++++++FORMAT OF TRAINED O/P FILE +++++++++++++++++++++
String(Mean_Shape)-Length
Mean_Shape
(int)Mean_Shape Size
Mean_Shape Points
String(Pixel_Coordinates)-Length
Pixel_coordinates
For Each Pixel Coordinate
    (int)Size of Pixel Coordinate Vector
    Pixel Coordinate Points
String(Cascade_Depth)-Length
Cascade_Depth
(int)CascadeDepth
String(Num_Trees_per_Cascade)-Length
Num_Trees_per_Cascade
(int)numTreesperCascade
For Each Cascade
    For Each Tree in Cascade
        (int)Tree Number
        For Each Split in Tree
            String(Split_Feature)-Length
            Split_Feature
            (int)Split Feature values
        For Each leaf in Tree
            String(Leaf)-Length
            Leaf
            (int)Leaf Values Vector
++++++++++++++++++++++++++++++++FORMAT OF TRAINED O/P FILE ++++++++++++++++++++++
*/

void KazemiFaceAlignImpl::writeSplitxml(FileStorage& fs, vector<splitFeature>& split)
{
    for(unsigned long i = 0; i<split.size();i++)
    {
        fs << "splitFeature" << "{";
        fs << "index1" << int(split[i].idx1);
        fs << "index2" << int(split[i].idx2);
        fs << "thresh" << int(split[i].thresh);
        fs << "}";
    }
}

void KazemiFaceAlignImpl::writeLeafxml( FileStorage& fs, vector< vector<Point2f> >& leaves)
{
    for (unsigned long j = 0; j < leaves.size(); ++j)
    {
        fs << "leaf" << "{";
        for(unsigned long i = 0 ; i < leaves[j].size() ; i++)
        {
            string attributeX = "x" + to_string(i);
            fs << attributeX << leaves[j][i].x;
            string attributeY = "y" + to_string(i);
            fs << attributeY << leaves[j][i].y;
        }
        fs << "}";
    }
}

void KazemiFaceAlignImpl::writeTreexml( FileStorage& fs, regressionTree& tree,unsigned long treeNo)
{
    string attributeTree = "tree" + to_string(treeNo);
    fs << attributeTree << "{";
    writeSplitxml(fs,tree.split);
    writeLeafxml(fs,tree.leaves);
    fs << "}";
}

void KazemiFaceAlignImpl::writeCascadexml( FileStorage& fs,vector<regressionTree>& forest)
{
    string attributeCascade = "cascade" + to_string(cascadeDepth);
    fs << attributeCascade << "{";
    for(int i=0;i<forest.size();i++){
        writeTreexml(fs,forest[i],i);
    }
    fs << "}";
}


}
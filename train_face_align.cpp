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
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

namespace cv{

bool KazemiFaceAlignImpl::trainCascade(std::map<string, vector<Point2f>>& landmarks, string path_prefix, CascadeClassifier& cascade)
{
    vector<trainSample> samples;
    vector< vector<Point2f> > pixelCoordinates;
    generateTestCoordinates(pixelCoordinates);
    fillData(samples, landmarks, path_prefix, cascade);
    FileStorage fs("194_Face_Align_Landmarks_Cascade.xml", FileStorage::WRITE);
    if (!fs.isOpened())
    {
        cerr << "Cannot open xml to save the model"<< endl;
        return false;
    }
    vector< vector<regressionTree> > cascadeFinal;
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        vector<Point2f> pixrel(pixelCoordinates[i].size());
        for (unsigned long j = 0; j < samples.size(); ++j)
        {
            pixrel = pixelCoordinates[i];
            calcRelativePixels(samples[j].currentShape,pixrel);
            extractPixelValues(samples[j],pixrel);
        }
        vector<regressionTree> forest = gradientBoosting(samples, pixelCoordinates[i]);
        cascadeFinal.push_back(forest);
        cout<<"Fitted "<< i + 1 <<"th regressor"<<endl;
        writeCascade(fs,forest,(i+1));
    }
    fs.release();
    return true;
}


bool KazemiFaceAlignImpl::fillData(vector<trainSample>& samples,std::map<string, vector<Point2f>>& landmarks,
                                    string path_prefix, CascadeClassifier& cascade)
{
    unsigned long currentCount =0;
    samples.resize(6*oversamplingAmount);
    for (unsigned long i = 0; i < oversamplingAmount; ++i)
    {
        int db = 0;
        for (map<string, vector<Point2f>>::iterator dbIterator = landmarks.begin();
            dbIterator != landmarks.end(); ++dbIterator)
        {
            if(db > 5)
                break;
            if(db == 0)
                {
                    //Assuming the current Shape of each sample's first initialization to be mean shape
                    samples[currentCount].img = getImage(dbIterator->first,path_prefix);
                    samples[currentCount].rect = faceDetector(samples[currentCount].img, cascade);
                    samples[currentCount].targetShape = dbIterator->second;
                    samples[currentCount].currentShape = meanShape;
                    samples[currentCount].residualShape  = calcDiff(samples[currentCount].currentShape, samples[currentCount].targetShape);
                }
            else
                {
                    //Assign some random image from the training sample as current shape
                    RNG number(getTickCount());
                    unsigned long randomIndex = (unsigned long)number.uniform(0, landmarks.size()-1);
                    samples[currentCount].img = getImage(dbIterator->first,path_prefix);
                    samples[currentCount].rect = faceDetector(samples[currentCount].img, cascade);
                    samples[currentCount].targetShape = dbIterator->second;
                    map<string, vector<Point2f>>::iterator item = landmarks.begin();
                    advance(item, randomIndex);
                    samples[currentCount].currentShape = item->second;
                    samples[currentCount].residualShape  = calcDiff(samples[currentCount].currentShape, samples[currentCount].targetShape);

                }
                db++;
                ++currentCount;
        }
    }
    cout<<currentCount<<": Training Samples Loaded.."<<endl;
    return true;
}

bool KazemiFaceAlignImpl::generateTestCoordinates(vector< vector<Point2f> >& pixelCoordinates)
{
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        vector<Point2f> testCoordinates;
        RNG rng(time(0));
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

bool KazemiFaceAlignImpl::calcRelativePixels(vector<Point2f>& sample,vector<Point2f>& pixelCoordinates)
{
    if(sample.size()!=meanShape.size()){
        String error_message = "Error while finding relative shape. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    double sampleShapeRectminx, sampleShapeRectminy, sampleShapeRectmaxx, sampleShapeRectmaxy;
    double sampleX[sample.size()] , sampleY[sample.size()];
    int pointcount=0;
    for (vector<Point2f>::iterator it = sample.begin(); it != sample.end(); ++it)
    {
        sampleX[pointcount] = (*it).x;
        sampleY[pointcount] = (*it).y;
        pointcount++;
    }
    sampleShapeRectminx = *min_element(sampleX , sampleX + sample.size());
    sampleShapeRectmaxx = *max_element(sampleX , sampleX + sample.size());
    sampleShapeRectminy = *min_element(sampleY , sampleY + sample.size());
    sampleShapeRectmaxy = *max_element(sampleY , sampleY + sample.size());
    Point2f sampleRefPoints[3];
    sampleRefPoints[0] = Point2f(sampleShapeRectminx , sampleShapeRectminy );
    sampleRefPoints[1] = Point2f( sampleShapeRectmaxx, sampleShapeRectminy );
    sampleRefPoints[2] = Point2f( sampleShapeRectminx, sampleShapeRectmaxy );
    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( meanShapeReferencePoints, sampleRefPoints);
    for(unsigned long i=0;i<pixelCoordinates.size();i++)
    {
        unsigned long in = findNearestLandmark(pixelCoordinates[i]);
        Point2f pt = pixelCoordinates[i] - meanShape[in];
        Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
        Mat D =warp_mat*C;
        pt.x=float(abs(D.at<double>(0,0)));
        pt.y=float(abs(D.at<double>(1,0)));
        pixelCoordinates[i]=pt+sample[in];
    }
    return true;
}

void KazemiFaceAlignImpl::writeSplit(FileStorage& fs, vector<splitFeature>& split)
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

void KazemiFaceAlignImpl::writeLeaf( FileStorage& fs, vector< vector<Point2f> >& leaves)
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

void KazemiFaceAlignImpl::writeTree( FileStorage& fs, regressionTree& tree,unsigned long treeNo)
{
    string attributeTree = "tree" + to_string(treeNo);
    fs << attributeTree << "{";
    writeSplit(fs,tree.split);
    writeLeaf(fs,tree.leaves);
    fs << "}";
}

void KazemiFaceAlignImpl::writeCascade( FileStorage& fs,vector<regressionTree>& forest,unsigned long cascadeDepth)
{
    string attributeCascade = "cascade" + to_string(cascadeDepth);
    fs << attributeCascade << "{";
    for(int i=0;i<forest.size();i++){
        writeTree(fs,forest[i],i);
    }
    fs << "}";
}

}
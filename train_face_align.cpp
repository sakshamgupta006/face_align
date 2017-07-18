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

bool KazemiFaceAlignImpl::trainCascade(std::map<string, vector<Point2f>>& landmarks, string path_prefix, CascadeClassifier& cascade, string outputName)
{
    vector<trainSample> samples;
    vector< vector<Point2f> > pixelCoordinates;
    generateTestCoordinates(pixelCoordinates);
    fillData(samples, landmarks, path_prefix, cascade);
    ofstream fs(outputName, ios::out | ios::binary);
    FileStorage fs2("194_Face_Align_Landmarks_Cascade.xml", FileStorage::WRITE);
    if (!fs.is_open())
    {
        cerr << "Cannot open xml to save the model"<< endl;
        return false;
    }
    vector< vector<regressionTree> > cascadeFinal;
    cout<<"Training Started"<<endl;
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        vector<Point2f> pixrel(pixelCoordinates[i].size());
        pixrel = pixelCoordinates[i];
        for (unsigned long j = 0; j < samples.size(); ++j)
        {
            calcRelativePixels(samples[j].currentShape,pixrel);
            extractPixelValues(samples[j],pixrel);
        }
        vector<regressionTree> forest = gradientBoosting(samples, pixelCoordinates[i]);
        cascadeFinal.push_back(forest);
        cout<<"Fitted "<< i + 1 <<"th regressor"<<endl;
        //writeCascadexml(fs2, forest);
    }
    writeModel(fs,cascadeFinal, pixelCoordinates);
    fs2.release();
    fs.close();
    displayresults2(samples);
    // Mat img = imread("../data/train/213033657_1.jpg");
    // imshow("InputImage", img);
    // testnewImage(img, cascadeFinal, pixelCoordinates, cascade);
    return true;
}

bool KazemiFaceAlignImpl::displayresults2(vector<trainSample>& samples)
{
    for (int i = 0; i < samples.size(); ++i)
     {
        for (int j = 0; j < samples[i].currentShape.size() ; ++j)
        {
            circle(samples[i].img, Point(samples[i].currentShape[j]), 2, Scalar(255,0,0) ,-1);
        }
        imshow("Results", samples[i].img);
        waitKey(0);
    }
    return true;
}

bool KazemiFaceAlignImpl::displayresults(trainSample& samples)
{
    for (int j = 0; j < samples.currentShape.size() ; ++j)
    {
        circle(samples.img, Point(samples.currentShape[j]), 2, Scalar(255,0,0), -1 );
    }
    imshow("Results", samples.img);
    waitKey(0);
    return true;
}

void KazemiFaceAlignImpl::testnewImage(Mat& image, vector< vector<regressionTree> >& cascadeFinal, vector< vector<Point2f>>& pixelCoordinates, CascadeClassifier& cascade)
{
    vector< vector<Point2f> > resultPoints;
    trainSample sample;
    sample.img = image;
    sample.rect = faceDetector(image, cascade);
    //sample.currentShape = getRelativeShapetoMean(sample, meanShape);
    getRelativeShapefromMean(sample, meanShape);
    //displayresults(sample);
        for (int i = 0; i < cascadeFinal.size() ; ++i)
        {
            vector<Point2f> pixel_relative = pixelCoordinates[i];
            calcRelativePixels(sample.currentShape, pixel_relative);
            extractPixelValues(sample, pixel_relative);
            for(unsigned long j = 0; j < cascadeFinal[i].size(); j++)
            {
                unsigned long k =0 ;
                while(k < cascadeFinal[i][j].split.size())
                {
                    if ((float)sample.pixelValues[cascadeFinal[i][j].split[k].idx1] - (float)sample.pixelValues[cascadeFinal[i][j].split[k].idx2] > cascadeFinal[i][j].split[k].thresh)
                        k = leftChild(k);
                    else
                        k = rightChild(k);
                }
                k = k - cascadeFinal[i][j].split.size();
                vector<Point2f> temp;
                temp.resize(sample.currentShape.size());
                for (unsigned long l = 0; l < sample.currentShape.size(); ++l)
                {
                    temp[l] = learningRate * cascadeFinal[i][j].leaves[k][l];
                }
                sample.currentShape = calcDiff(temp, sample.currentShape);
            }
        }
    displayresults(sample);
}

bool KazemiFaceAlignImpl::fillData(vector<trainSample>& samples,std::map<string, vector<Point2f>>& landmarks,
                                    string path_prefix, CascadeClassifier& cascade)
{
    unsigned long currentCount =0;
    samples.resize(301*oversamplingAmount);
    int db = 0;
    for (map<string, vector<Point2f>>::iterator dbIterator = landmarks.begin();
            dbIterator != landmarks.end(); ++dbIterator)
    {
        unsigned int firstCount = 0;
        if(db > 300)
            break;
        for (unsigned long i = 0; i < oversamplingAmount; ++i)
        {
            if(i == 0)
            {
                //Assuming the current Shape of each sample's first initialization to be mean shape
                samples[currentCount].img = getImage(dbIterator->first,path_prefix);
                samples[currentCount].rect = faceDetector(samples[currentCount].img, cascade);
                if(samples[currentCount].rect.size() != 1)
                {
                    samples.erase(samples.begin() + currentCount);
                    continue;
                }
                samples[currentCount].targetShape = dbIterator->second;
                getRelativeShapefromMean(samples[currentCount], meanShape);
                samples[currentCount].residualShape  = calcDiff(samples[currentCount].currentShape, samples[currentCount].targetShape);
                if(samples[currentCount].currentShape.size() != 0)
                {
                    firstCount = currentCount;
                    currentCount++;
                }
            }
            else
            {
                //Assign some random image from the training sample as current shape
                RNG number(getTickCount());
                unsigned long randomIndex = (unsigned long)number.uniform(0, landmarks.size()-1);
                samples[currentCount].img = samples[firstCount].img;
                samples[currentCount].rect = samples[firstCount].rect;
                if(samples[currentCount].rect.size() != 1)
                {
                    samples.erase(samples.begin() + currentCount);
                    continue;
                }
                samples[currentCount].targetShape = samples[firstCount].targetShape;
                map<string, vector<Point2f>>::iterator item = landmarks.begin();
                advance(item, randomIndex);
                samples[currentCount].currentShape = item->second;
                if(samples[currentCount].currentShape.size() != 0)
                {
                    getRelativeShape(samples[currentCount]);
                    samples[currentCount].residualShape  = calcDiff(samples[currentCount].currentShape, samples[currentCount].targetShape);
                    currentCount++;
                }
            }
        }
    db++;
    }
    samples.erase(samples.begin()+currentCount-1, samples.end());
    cout<<"Sample size"<<samples.size()<<endl;
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
    if(sample.size()!= meanShape.size())
    {
        String errmsg = "Shape Size Mismatch Detected";
        CV_Error(Error::StsBadArg, errmsg);
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
    Mat affineMatrix = getAffineTransform( meanShapeReferencePoints, sampleRefPoints);
    for(unsigned long i=0;i<pixelCoordinates.size();i++)
    {
        unsigned long in = findNearestLandmark(pixelCoordinates[i]);
        Point2f point = pixelCoordinates[i] - meanShape[in];
        Mat fiducialPointMat = (Mat_<double>(3,1) << point.x, point.y, 1);
        Mat resultAffineMat = affineMatrix * fiducialPointMat;
        point.x = float(abs(resultAffineMat.at<double>(0,0)));
        point.y = float(abs(resultAffineMat.at<double>(1,0)));
        pixelCoordinates[i] = point + sample[in];
    }
    return true;
}

bool KazemiFaceAlignImpl::getRelativeShape(trainSample& sample)
{
    if(sample.targetShape.size()!= sample.currentShape.size())
    {
        String error_message = "Shape Mismatch Encountered";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    double samplecurrentShapeRectminx, samplecurrentShapeRectminy, samplecurrentShapeRectmaxx, samplecurrentShapeRectmaxy;
    double samplecurrentX[sample.currentShape.size()] , samplecurrentY[sample.currentShape.size()];
    int pointcount=0;
    for (vector<Point2f>::iterator it = sample.currentShape.begin(); it != sample.currentShape.end(); ++it)
    {
        samplecurrentX[pointcount] = (*it).x;
        samplecurrentY[pointcount] = (*it).y;
        pointcount++;
    }
    samplecurrentShapeRectminx = *min_element(samplecurrentX , samplecurrentX + sample.currentShape.size());
    samplecurrentShapeRectmaxx = *max_element(samplecurrentX , samplecurrentX + sample.currentShape.size());
    samplecurrentShapeRectminy = *min_element(samplecurrentY , samplecurrentY + sample.currentShape.size());
    samplecurrentShapeRectmaxy = *max_element(samplecurrentY , samplecurrentY + sample.currentShape.size());
    Point2f samplecurrentRefPoints[3];
    samplecurrentRefPoints[0] = Point2f( samplecurrentShapeRectminx , samplecurrentShapeRectminy );
    samplecurrentRefPoints[1] = Point2f( samplecurrentShapeRectmaxx, samplecurrentShapeRectminy );
    samplecurrentRefPoints[2] = Point2f( samplecurrentShapeRectminx, samplecurrentShapeRectmaxy );

    double sampletargetShapeRectminx, sampletargetShapeRectminy, sampletargetShapeRectmaxx, sampletargetShapeRectmaxy;
    double sampletargetX[sample.targetShape.size()] , sampletargetY[sample.targetShape.size()];
    int pointcount2=0;
    for (vector<Point2f>::iterator it = sample.targetShape.begin(); it != sample.targetShape.end(); ++it)
    {
        sampletargetX[pointcount2] = (*it).x;
        sampletargetY[pointcount2] = (*it).y;
        pointcount2++;
    }
    sampletargetShapeRectminx = *min_element(sampletargetX , sampletargetX + sample.targetShape.size());
    sampletargetShapeRectmaxx = *max_element(sampletargetX , sampletargetX + sample.targetShape.size());
    sampletargetShapeRectminy = *min_element(sampletargetY , sampletargetY + sample.targetShape.size());
    sampletargetShapeRectmaxy = *max_element(sampletargetY , sampletargetY + sample.targetShape.size());
    Point2f sampletargetRefPoints[3];
    sampletargetRefPoints[0] = Point2f( sampletargetShapeRectminx , sampletargetShapeRectminy );
    sampletargetRefPoints[1] = Point2f( sampletargetShapeRectmaxx, sampletargetShapeRectminy );
    sampletargetRefPoints[2] = Point2f( sampletargetShapeRectminx, sampletargetShapeRectmaxy );

    Mat affineMatrix = getAffineTransform( samplecurrentRefPoints, sampletargetRefPoints );
    for (vector<Point2f>::iterator it = sample.currentShape.begin(); it !=sample.currentShape.end(); it++)
    {
        Point2f point = (*it);
        Mat fiducialPointMat = (Mat_<double>(3,1) << point.x, point.y, 1);
        Mat resultAffineMat = affineMatrix * fiducialPointMat;
        point.x = float(abs(resultAffineMat.at<double>(0,0)));
        point.y = float(abs(resultAffineMat.at<double>(1,0)));
        (*it) = point;
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
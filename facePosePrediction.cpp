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

namespace cv
{

bool KazemiFaceAlignImpl::loadTrainedModel(ifstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> >& pixelCoordinates)
{
    //START READING MEAN SHAPE
    size_t stringLength;
    fs.read((char*)&stringLength, sizeof(size_t));
    string meanShapestring;
    meanShapestring.resize(stringLength);
    fs.read(&meanShapestring[0], stringLength);
    if(meanShapestring != "Mean_Shape")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    unsigned long meanShapesize;
    fs.read((char*)&meanShapesize, sizeof(meanShapesize));
    meanShape.resize(meanShapesize);
    fs.read((char*)&meanShape[0], sizeof(Point2f)*meanShapesize);
    //END MEANSHAPE READING
    //START READING PIXEL COORDINATES
    fs.read((char*)&stringLength, sizeof(size_t));
    string pixelCoordinatesstring;
    pixelCoordinatesstring.resize(stringLength);
    fs.read(&pixelCoordinatesstring[0], stringLength);
    if(pixelCoordinatesstring != "Pixel_Coordinates")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    unsigned long pixelCoordinatesMainsize;
    fs.read((char*)&pixelCoordinatesMainsize, sizeof(pixelCoordinatesMainsize));
    pixelCoordinates.resize(pixelCoordinatesMainsize);
    for (unsigned long i = 0; i < pixelCoordinatesMainsize; ++i)
    {
        unsigned long pixelCoordinatesVecsize;
        fs.read((char*)&pixelCoordinatesVecsize, sizeof(pixelCoordinatesVecsize));
        pixelCoordinates[i].resize(pixelCoordinatesVecsize);
        fs.read((char*)&pixelCoordinates[i][0], sizeof(Point2f) * pixelCoordinatesVecsize);
    }
    //END PIXEL COORDINATES READING
    //START READING THE FOREST
    fs.read((char*)&stringLength, sizeof(size_t));
    string cascadedepthstring;
    cascadedepthstring.resize(stringLength);
    fs.read(&cascadedepthstring[0], stringLength);
    if(cascadedepthstring != "Cascade_Depth")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    fs.read((char*)&cascadeDepth, sizeof(cascadeDepth));
    fs.read((char*)&stringLength, sizeof(size_t));
    string numtresspercascadestring;
    numtresspercascadestring.resize(stringLength);
    fs.read(&numtresspercascadestring[0], stringLength);
    if(numtresspercascadestring != "Num_Trees_per_Cascade")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    fs.read((char*)&numTreesperCascade, sizeof(numTreesperCascade));
    forest.resize(cascadeDepth);
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        forest[i].resize(numTreesperCascade);
        for (unsigned long j = 0; j < numTreesperCascade; ++j)
        {
            unsigned long splitsize;
            fs.read((char*)&splitsize, sizeof(splitsize));
            forest[i][j].split.resize(splitsize);
            for (unsigned long k = 0; k < splitsize; ++k)
            {
                fs.read((char*)&stringLength, sizeof(size_t));
                string splitstring;
                splitstring.resize(stringLength);
                fs.read(&splitstring[0], stringLength);
                if(splitstring != "Split_Feature")
                {
                    String errmsg = "Model Not Saved Properly";
                    CV_Error(Error::StsBadArg, errmsg);
                    return false;
                }
                splitFeature temp;
                fs.read((char*)&temp, sizeof(splitFeature));
                forest[i][j].split[k] = temp;
            }
            unsigned long leavessize;
            fs.read((char*)&leavessize, sizeof(leavessize));
            forest[i][j].leaves.resize(leavessize);
            for (unsigned long k = 0; k < leavessize; ++k)
            {
                fs.read((char*)&stringLength, sizeof(size_t));
                string leafstring;
                leafstring.resize(stringLength);
                fs.read(&leafstring[0], stringLength);
                if(leafstring != "Leaf")
                {
                    String errmsg = "Model Not Saved Properly";
                    CV_Error(Error::StsBadArg, errmsg);
                    return false;
                }
                unsigned long leafsize;
                fs.read((char*)&leafsize, sizeof(leafsize));
                forest[i][j].leaves[k].resize(leafsize);
                fs.read((char*)&forest[i][j].leaves[k][0], sizeof(Point2f) * leafsize);
            }
        }
    }
//FOREST READING END
    return true;
}

vector< vector<Point2f> > KazemiFaceAlignImpl::getFacialLandmarks(Mat& image, vector< vector<regressionTree> >& cascadeFinal, vector< vector<Point2f>>& pixelCoordinates, CascadeClassifier& cascade)
{
    vector< vector<Point2f> > resultPoints;
    trainSample sample;
    sample.img = image;
    sample.rect = faceDetector(image, cascade);
    sample.currentShape = getRelativeShapetoMean(sample, meanShape);
    vector<Point2f> othervec;
    othervec.resize(sample.currentShape.size());
    othervec = sample.currentShape;
    for(unsigned int l = 0; l < sample.rect.size(); l++)
    {
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
                temp.resize(cascadeFinal[i][j].leaves[k].size());
                for (int g = 0; g < cascadeFinal[i][j].leaves[k].size(); ++g)
                {
                    temp[g] = learningRate * cascadeFinal[i][j].leaves[k][g];
                }
                sample.currentShape = calcDiff(temp, sample.currentShape);
                othervec = calcDiff(cascadeFinal[i][j].leaves[k], othervec);
            }
        }
        resultPoints[l] = sample.currentShape;
    }
    //change back to original coordinates
    sample.currentShape = getRelativeShapefromMean(sample, sample.currentShape);
    displayresults(sample);
    return resultPoints;
}



}
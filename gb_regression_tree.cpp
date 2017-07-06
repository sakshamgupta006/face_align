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
#include <vector>

using namespace std;
using namespace cv;

namespace cv{

double KazemiFaceAlignImpl::getDistance(Point2f first , Point2f second)
{
    return sqrt(pow((first.x - second.x),2) + pow((first.y - second.y),2));
}

splitFeature KazemiFaceAlignImpl::randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates)
{
    splitFeature feature;
    double acceptProbability, randomdoublenumber;
    unsigned long attempts = 15;
    do
    {
        RNG rnd(getTickCount());
        feature.idx1 = rnd.uniform(0,numFeature);
        feature.idx2 = rnd.uniform(0,numFeature);
        double dist = getDistance(pixelCoordinates[feature.idx1] , pixelCoordinates[feature.idx2]);
        acceptProbability = exp(-dist/lambda);
        randomdoublenumber = rnd.uniform(0.,1.);
        feature.thresh = rnd.uniform(double(-255),double(255)); //Check Validity
        attempts--;
    }
    while((feature.idx1 == feature.idx2 || randomdoublenumber > acceptProbability) && (attempts > 0));
    return feature;
}

splitFeature KazemiFaceAlignImpl::splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long start ,
                                                unsigned long end, vector<Point2f>& sum, vector<Point2f>& leftSum, vector<Point2f>& rightSum )
{
    vector< vector<Point2f> > leftSums;
    leftSums.resize(numTestSplits);
    vector<splitFeature> features;
    for (unsigned int i = 0; i < numTestSplits; ++i)
    {
        features.push_back(randomSplitFeatureGenerator(pixelCoordinates));
        leftSums[i].resize(samples[0].targetShape.size());
    }
    vector<unsigned long> leftCount;
    leftCount.resize(numTestSplits);
    ///////--------------------SCOPE OF THREADING----------------------/////////////
    for (unsigned long i = start; i < end ; ++i)
    {
        for (unsigned long j = 0; j < numTestSplits; ++j)
        {
            if((float)samples[i].pixelValues[features[j].idx1] - (float)samples[i].pixelValues[features[j].idx2] > features[j].thresh)
            {
                leftSums[j] = calcSum(leftSums[j], samples[i].residualShape);
                leftCount[j] += 1;
            }
        }
    }
    ///////--------------------SCOPE OF THREADING----------------------/////////////
    //Select the best feature
    double bestScore = -1;
    unsigned long bestFeature = 0;
    vector<Point2f> tempFeature;
    tempFeature.resize(sum.size());
    for (unsigned long i = 0; i < numTestSplits; ++i)
    {
        double currentScore = 0;
        unsigned long rightCount = end - start + 1 - leftCount[i];  // See for the extra 1
        if(leftCount[i] != 0 && rightCount != 0)
        {
            tempFeature = calcDiff(sum, leftSums[i]);
            //To calculate score
            Point2f leftSumsDot(0,0), tempFeatureDot(0,0);
            for (unsigned long dotit = 0; dotit < leftSums.size(); ++dotit)
            {
                leftSumsDot.x += (double)pow((leftSums[i][dotit].x),2);
                leftSumsDot.y += (double)pow((leftSums[i][dotit].y),2);
                tempFeatureDot.x += (double)pow((tempFeature[dotit].x),2);
                tempFeatureDot.y += (double)pow((tempFeature[dotit].y),2);
            }
            double leftSumsDotRes = sqrt(pow(leftSumsDot.x, 2) + pow(leftSumsDot.y, 2));
            double tempFeatureDotRes = sqrt(pow(tempFeatureDot.x, 2) + pow(tempFeatureDot.y, 2));
            currentScore = leftSumsDotRes/leftCount[i] + tempFeatureDotRes/rightCount;
            if(currentScore > bestScore)
            {
                bestScore = currentScore;
                bestFeature = i;
            }
        }
    }
    leftSum = leftSums[bestFeature];
    if(leftSum.size() != 0)
        rightSum = calcDiff(sum, leftSum);
    else
    {
        rightSum = sum;
        leftSum.assign(sum.size(), Point2f(0,0));
    }
return features[bestFeature];
}

regressionTree KazemiFaceAlignImpl::buildRegressionTree(vector<trainSample>& samples, vector<Point2f>& pixelCoordinates)
{
    regressionTree tree;
    //partition queue will store the extent of leaf nodes
    deque< pair<unsigned long, unsigned long > >  partition;
    partition.push_back(make_pair(0, (unsigned long)samples.size()));
    const unsigned long numSplitNodes = (unsigned long)(pow(2 , (double)getTreeDepth()) - 1);
    vector< vector<Point2f> >sums(numSplitNodes*2 + 1);
    ////---------------------------------SCOPE OF THREADING---------------------------------------////
    //Initialize Sum for the root node
    for (unsigned long i = 0; i < samples.size(); i++)
    {
        sums[0] = calcSum(sums[0],samples[i].residualShape);
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
    //following Dlib's approach
    vector<Point2f> residualSum(samples[0].targetShape.size());
    tree.leaves.resize(partition.size());
    //Use partition value to calculate average value of leafs
    for (unsigned long int i = 0; i < partition.size(); ++i)
    {
        unsigned long currentCount = partition[i].second - partition[i].first + 1;
        for (unsigned long j = partition[i].first; j < partition[i].second; ++j)
            residualSum = calcSum(residualSum, samples[j].currentShape);
        //Calculate Reciprocal
        for (unsigned long k = 0; k < residualSum.size(); ++k)
        {
            if(residualSum[k].x != 0)
                residualSum[k].x  = 1 / residualSum[k].x;
            else
                residualSum[k].x = 0;
            if(residualSum[k].y != 0)
                residualSum[k].y = 1 / residualSum[k].y;
            else
                residualSum[k].y = 0;
        }
        //End Reciprocal
        if(partition[i].first != partition[i].second)
            {
                //tree.leaves[i] = calcMul(residualSum,sums[numSplitNodes]); // To be fully implemented in case of missing labels
                tree.leaves[i] = sums[numSplitNodes+i];
                for (unsigned long l = 0; l < residualSum.size(); ++l)
                {
                    tree.leaves[i][l] = learningRate * tree.leaves[i][l];
                }
            }
        else
            tree.leaves[i].assign(samples[0].targetShape.size(), Point2f(0,0));
        vector<Point2f> tempvector;
        tempvector.resize(samples[0].targetShape.size());
        for (unsigned long j = partition[i].first; j < partition[i].second; ++j)
        {
            tempvector = calcSum(samples[j].currentShape, tree.leaves[i]);
            samples[j].currentShape = calcDiff(samples[j].targetShape, tempvector);
            //To be fully implemented in case of missing labels
            /* for (unsigned long m = 0; m < samples[j].currentShape.size(); ++m)
            {
                if(samples[j].residualShape[m].x == 0 &&  samples[j].residualShape[m].y == 0)
                    samples[j].targetShape[m] = samples[j].currentShape[m];
            } */
        }
    }
    return tree;
}

vector<regressionTree> KazemiFaceAlignImpl::gradientBoosting(vector<trainSample>& samples, vector<Point2f>& pixelCoordinates)
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

unsigned long KazemiFaceAlignImpl::partitionSamples(splitFeature split, vector<trainSample>& samples, unsigned long start, unsigned long end)
{
    unsigned long initial = start;
    for (unsigned long j = start; j < end; j++)
    {
        if((float)samples[j].pixelValues[split.idx1] - (float)samples[j].pixelValues[split.idx2] > split.thresh)
        {
            swap(samples[initial],samples[j]);
            initial++;
        }
    }
    return initial;
}

}

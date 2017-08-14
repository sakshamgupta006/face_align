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

using namespace std;
using namespace cv;

namespace cv
{

//Parallelization Inducing Functions
class calcSumSample : public ParallelLoopBody
{
public:
    calcSumSample (vector<trainSample>& samples, vector<Point2f>& sumOutput)
        : _samples(samples), _sumOutput(sumOutput)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        for (unsigned long r = range.start; r < range.end; r++)
        {
            for (unsigned long i = 0; i < _samples[r].currentShape.size(); ++i)
            {
                _samples[r].currentShape[i] = _samples[r].currentShape[i] + _sumOutput[i];
            }
        }
    }

private:
    vector<trainSample>& _samples;
    vector<Point2f>& _sumOutput;
};

class calcDiffSample : public ParallelLoopBody
{
public:
    calcDiffSample (vector<trainSample>& samples, vector<Point2f>& sumOutput)
        : _samples(samples), _sumOutput(sumOutput)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        for (unsigned long r = range.start; r < range.end; r++)
        {
            for (unsigned long i = 0; i < _samples[r].targetShape.size(); ++i)
            {
                _samples[r].residualShape[i] = _samples[r].targetShape[i] - _samples[r].currentShape[i];
                _sumOutput[i] = _samples[r].residualShape[i] + _sumOutput[i];
            }
        }
    }

private:
    vector<trainSample>& _samples;
    vector<Point2f>& _sumOutput;
};

class calcConstraintSum : public ParallelLoopBody, protected KazemiFaceAlignImpl
{
public:
    calcConstraintSum(vector<trainSample> &samples, vector< vector<Point2f> >& leftSums, vector<splitFeature>& features, vector<unsigned long>& leftCount)
        : _samples(samples), _leftSums(leftSums), _features(features), _leftCount(leftCount)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        for (unsigned long r = range.start; r < range.end ; r++)
        {
            for (unsigned long j = 0; j < numTestSplits; ++j)
            {
                if((float)_samples[r].pixelValues[_features[j].idx1] - (float)_samples[r].pixelValues[_features[j].idx2] > 
                    _features[j].thresh)
                {
                    for(unsigned long m = 0; m < _samples[r].residualShape.size(); m++)
                    {
                        _leftSums[j][m] += _samples[r].residualShape[m];
                    }
                    _leftCount[j] += 1;
                }
            } 
        }
    }
private:
    vector<trainSample>& _samples;
    vector< vector<Point2f> >& _leftSums;
    vector<splitFeature>& _features;
    vector<unsigned long>& _leftCount;
};

double KazemiFaceAlignImpl::getDistance(Point2f first , Point2f second)
{
    return sqrt(pow((first.x - second.x),2) + pow((first.y - second.y),2));
}

splitFeature KazemiFaceAlignImpl::randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates)
{
    splitFeature feature;
    double acceptProbability, randomdoublenumber;
    RNG rnd(getTickCount());
    do
    {
        feature.idx1 = rnd.uniform(0,numFeature);
        feature.idx2 = rnd.uniform(0,numFeature);
        double dist = getDistance(pixelCoordinates[feature.idx1] , pixelCoordinates[feature.idx2]);
        acceptProbability = exp(-dist/lambda);
        randomdoublenumber = rnd.uniform(0.,1.);
    }
    while((feature.idx1 == feature.idx2 || !(acceptProbability > randomdoublenumber)));
    feature.thresh = (rnd.uniform(0.,1.)*255);// - 128.0) / 2.0 ;
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
        leftSums[i].resize(numLandmarks);
    }
    vector<unsigned long> leftCount;
    leftCount.resize(numTestSplits);
    parallel_for_(Range(start, end), calcConstraintSum(samples, leftSums, features, leftCount));
    //Select the best feature
    double bestScore = -1;
    double score = -1;
    unsigned long bestFeature = 0;
    vector<Point2f> rightSums,leftSumTree;
    rightSums.resize(sum.size());
    leftSumTree.resize(sum.size());
    for (unsigned long i = 0; i < numTestSplits; ++i)
    {
        double currentScore = 0;
        unsigned long rightCount = (end - start + 1) - leftCount[i];
        for (int j = 0; j < leftSums[i].size(); ++j)
        {
            if(rightCount != 0)
            {
                rightSums[j].x = (sum[j].x - leftSums[i][j].x)/ rightCount ;
                rightSums[j].y = (sum[j].y - leftSums[i][j].y)/ rightCount ;
            }
            else
                rightSums[j] = Point2f(0,0);
            if(leftCount[i] != 0)
            {
                leftSumTree[j].x = leftSums[i][j].x / leftCount[i];
                leftSumTree[j].y = leftSums[i][j].y / leftCount[i];
            }
            else
                leftSumTree[j] = Point2f(0,0);   
        }
        Point2f point1(0,0);
        Point2f point2(0,0);
        for(unsigned long k = 0; k < leftSumTree.size(); k++)
        {
            point1.x += (float)pow(leftSumTree[k].x, 2);
            point1.y += (float)pow(leftSumTree[k].y, 2);
            point2.x += (float)pow(rightSums[k].x, 2);
            point2.y += (float)pow(rightSums[k].y, 2);
        }
        score = (double)sqrt(point1.x + point1.y)/leftCount[i] + (double)sqrt(point2.x + point2.y)/rightCount;
        if(score > bestScore)
        {
            bestScore = score;
            bestFeature = i;
        }
    }
    leftSums[bestFeature].swap(leftSum);
    rightSum.resize(sum.size());
    for(unsigned long k = 0; k < sum.size(); k++)
    {
        rightSum[k].x = sum[k].x - leftSum[k].x;
        rightSum[k].y = sum[k].y - leftSum[k].y;
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
    tree.split.resize(numSplitNodes);
    vector<Point2f> zerovector;
    zerovector.assign(numLandmarks, Point2f(0,0));
    vector< vector<Point2f> > sums(numSplitNodes*2 + 1, zerovector);
    //Initialize Sum for the root node
    parallel_for_(Range(0, samples.size()), calcDiffSample(samples, sums[0]));
    //Iteratively generate Splits in the samples
    for (unsigned long int i = 0; i < numSplitNodes; i++)
    {
        pair<unsigned long, unsigned long> rangeleaf = partition.front();
        partition.pop_front();
        splitFeature split = splitGenerator(samples, pixelCoordinates, rangeleaf.first, rangeleaf.second, sums[i], sums[leftChild(i)], sums[rightChild(i)]);
        tree.split[i] = split;
        unsigned long mid = partitionSamples(split, samples, rangeleaf.first, rangeleaf.second);
        partition.push_back(make_pair(rangeleaf.first, mid));
        partition.push_back(make_pair(mid, rangeleaf.second));
    }
    //following Dlib's approach
    vector<Point2f> residualSum(numLandmarks);
    tree.leaves.resize(partition.size());
    //Use partition value to calculate average value of leafs
    for (unsigned long int i = 0; i < partition.size(); ++i)
    {
        unsigned long currentCount = partition[i].second - partition[i].first + 1;
        if(partition[i].first != partition[i].second)
            {
                tree.leaves[i] = sums[numSplitNodes+i];
                for (unsigned long l = 0; l < tree.leaves[i].size(); ++l)
                {
                    tree.leaves[i][l].x = ( learningRate * tree.leaves[i][l].x ) / currentCount;
                    tree.leaves[i][l].y = (learningRate * tree.leaves[i][l].y ) / currentCount;
                }
            }
        else
            tree.leaves[i].assign(numLandmarks, Point2f(0,0));
        parallel_for_(Range(partition[i].first,partition[i].second), calcSumSample(samples, tree.leaves[i]));
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
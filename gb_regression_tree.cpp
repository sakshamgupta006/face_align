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
                _sumOutput[i] = _samples[r].currentShape[i] + _sumOutput[i];
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
    calcDiffSample (vector<trainSample>& samples)
        : _samples(samples)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        for (unsigned long r = range.start; r < range.end; r++)
        {
            for (unsigned long i = 0; i < _samples[r].targetShape.size(); ++i)
            {
                _samples[r].residualShape[i] = _samples[r].targetShape[i] - _samples[r].currentShape[i];
            }
        }
    }

private:
    vector<trainSample>& _samples;
};

class calcConstraintSum : public ParallelLoopBody, protected KazemiFaceAlignImpl
{
public:
    calcConstraintSum(vector<trainSample>& samples, vector< vector<Point2f> >& leftSums, vector<splitFeature>& features, vector<unsigned long>& leftCount)
        : _samples(samples), _leftSums(leftSums), _features(features), _leftCount(leftCount)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        for (unsigned long r = range.start; r < range.end ; r++)
        {
            for (unsigned long j = 0; j < numTestSplits; ++j)
            {
                if((float)_samples[r].pixelValues[_features[j].idx1] - (float)_samples[r].pixelValues[_features[j].idx2] > _features[j].thresh)
                {
                    //calcSum(leftSums[j], samples[i].residualShape, leftSums[j]);
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




class parallelRandomSplitFeatureGenerator : public ParallelLoopBody, public KazemiFaceAlignImpl
{
public:
    parallelRandomSplitFeatureGenerator (vector<Point2f>& pixelCoordinates, vector<splitFeature>& features)
        : _pixelCoordinates(pixelCoordinates), _features(features)
    {
    }

    virtual void operator ()(const Range& range) const
    {
        for (unsigned long r = range.start; r < range.end; r++)
        {
            splitFeature feature;
            double acceptProbability, randomdoublenumber;
            unsigned long attempts = 20;
            RNG rnd;
            rnd(getTickCount());
            do
            {
                feature.idx1 = (int)rnd.uniform(0,numFeature);
                feature.idx2 = (int)rnd.uniform(0,numFeature);
                //double dist = getDistance(pixelCoordinates[feature.idx1] , pixelCoordinates[feature.idx2]);
                double dist = sqrt( pow(_pixelCoordinates[feature.idx1].x - _pixelCoordinates[feature.idx2].x, 2) + pow(_pixelCoordinates[feature.idx1].y - _pixelCoordinates[feature.idx2].y, 2));
                acceptProbability = exp(-dist/lambda);
                randomdoublenumber = (double)rnd.uniform(0.,1.);
                attempts--;
            }
            while((feature.idx1 == feature.idx2 || !(acceptProbability > randomdoublenumber)));//&& (attempts > 0));
            feature.thresh = ((float)rnd.uniform(0.,1.)*256 - 128) / 2.0; //Check Validity
            //return feature;
            _features.push_back(feature);
        }
    }

private:
    vector<Point2f>& _pixelCoordinates;
    vector<splitFeature>& _features;
};





double KazemiFaceAlignImpl::getDistance(Point2f first , Point2f second)
{
    return sqrt(pow((first.x - second.x),2) + pow((first.y - second.y),2));
}

splitFeature KazemiFaceAlignImpl::randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates)
{
    splitFeature feature;
    double acceptProbability, randomdoublenumber;
    unsigned long attempts = 100;
    RNG rnd;
    rnd(getTickCount());
    do
    {
        feature.idx1 = rnd.uniform(0,numFeature);
        feature.idx2 = rnd.uniform(0,numFeature);
        double dist = sqrt( pow(pixelCoordinates[feature.idx1].x - pixelCoordinates[feature.idx2].x, 2) + pow(pixelCoordinates[feature.idx1].y - pixelCoordinates[feature.idx2].y, 2));
        acceptProbability = exp(-dist/lambda);
        //cout<<"dist"<<dist<<" prob"<<acceptProbability<<endl;
        randomdoublenumber = rnd.uniform(0.,1.);
        attempts--;
    }
    while( feature.idx1 == feature.idx2 || !(acceptProbability > randomdoublenumber));// && (attempts > 0));
    feature.thresh = (rnd.uniform(0.,1.)*256 - 128) / 2.0; //Check Validity
    //cout<<"Got a valid split"<<endl;
    return feature;
}

splitFeature KazemiFaceAlignImpl::splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long start ,
                                                unsigned long end, vector<Point2f>& sum, vector<Point2f>& leftSum, vector<Point2f>& rightSum )
{
    vector< vector<Point2f> > leftSums;
    leftSums.resize(numTestSplits);
    vector<splitFeature> features(numTestSplits);
    //parallel_for_(Range(0, numTestSplits), parallelRandomSplitFeatureGenerator(pixelCoordinates, features));
    double t = (double)getTickCount();
    for (unsigned int i = 0; i < numTestSplits; ++i)
    {
        features.push_back(randomSplitFeatureGenerator(pixelCoordinates));

        leftSums[i].resize(samples[0].targetShape.size());
    }
    t = (double)getTickCount() - t;
    cout<<"Time Taken to generate 20 splits = "<< t/(getTickFrequency()) <<" ms"<<endl;
    //cout<<"Random Splits Generated"<<endl;
    vector<unsigned long> leftCount;
    leftCount.resize(numTestSplits);
    //parallel_for_(Range(start, end), calcConstraintSum(samples, leftSums, features, leftCount));

    ///////--------------------SCOPE OF THREADING----------------------/////////////
    for (unsigned long i = start; i < end ; ++i)
    {
        for (unsigned long j = 0; j < numTestSplits; ++j)
        {
            if((float)samples[i].pixelValues[features[j].idx1] - (float)samples[i].pixelValues[features[j].idx2] > features[j].thresh)
            {
                calcSum(leftSums[j], samples[i].residualShape, leftSums[j]);
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
            calcDiff(sum, leftSums[i], tempFeature);
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
        calcDiff(sum, leftSum, rightSum);
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
    vector<Point2f> zerovector;
    zerovector.assign(samples[0].targetShape.size(), Point2f(0,0));
    vector< vector<Point2f> > sums(numSplitNodes*2 + 1 , zerovector);

    ////---------------------------------SCOPE OF THREADING---------------------------------------////
    //Parallel approach
    parallel_for_(Range(0, samples.size()), calcDiffSample(samples));
    parallel_for_(Range(0, samples.size()), calcSumSample(samples, sums[0]));
    //cout<<"Calculate diff and sum"<<endl;
    //Initialize Sum for the root node
    // for (unsigned long i = 0; i < samples.size(); i++)
    // {
    //     calcDiff(samples[i].targetShape, samples[i].currentShape, samples[i].residualShape);
    //     calcSum(samples[i].residualShape, sums[0], sums[0]);
    // }
    //Iteratively generate Splits in the samples
    for (unsigned long int i = 0; i < numSplitNodes; i++)
    {
        pair<unsigned long, unsigned long> rangeleaf = partition.front();
        partition.pop_front();
        splitFeature split = splitGenerator(samples, pixelCoordinates, rangeleaf.first, rangeleaf.second, sums[i], sums[leftChild(i)], sums[rightChild(i)]);
        tree.split.push_back(split);
        unsigned long mid = partitionSamples(split, samples, rangeleaf.first, rangeleaf.second);
        partition.push_back(make_pair(rangeleaf.first, mid));
        partition.push_back(make_pair(mid, rangeleaf.second));
    }
    //following Dlib's approach
    vector<Point2f> residualSum(samples[0].targetShape.size());
    tree.leaves.resize(partition.size());
    vector<Point2f> onesvector(samples[0].targetShape.size());
    onesvector.assign(samples[0].targetShape.size(), Point2f(1,1));
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
            tree.leaves[i].assign(samples[0].targetShape.size(), Point2f(0,0));
        vector<Point2f> tempvector;
        tempvector.resize(samples[0].targetShape.size());

        //parallel_for_(Range(partition[i].first,partition[i].second), calcSumSample(samples, tree.leaves[i]));
        for (unsigned long j = partition[i].first; j < partition[i].second; ++j)
        {
            calcSum(samples[j].currentShape, tree.leaves[i], samples[j].currentShape);
            //calcDiff(samples[j].targetShape, tempvector, samples[j].currentShape);
            //To be fully implemented in case of missing labels
            /* for (unsigned long m = 0; m < samples[j].currentShape.size(); ++m)
            {
                if(samples[j].residualShape[m].x == 0 &&  samples[j].residualShape[m].y == 0)
                    samples[j].targetShape[m] = samples[j].currentShape[m];
            }*/
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
    // for (unsigned long i = 0; i < samples.size(); i++)
    // {
    //     calcDiff(samples[i].targetShape, samples[i].residualShape, samples[i].currentShape);
    // }
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
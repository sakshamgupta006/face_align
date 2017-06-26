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
    vector< vector<Point2f> > leftSums;
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

regressionTree KazemiFaceAlignImpl::buildRegressionTree(vector<trainSample>& samples, vector<Point2f> pixelCoordinates)
{
    regressionTree tree;
    //partition queue will store the extent of leaf nodes
    deque< pair<unsigned long, unsigned long > >  partition;
    partition.push_back(make_pair(0, (unsigned long)samples.size()));
    const unsigned long numSplitNodes = (unsigned long)(pow(2 , (double)getTreeDepth()) - 1);
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

}

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

bool KazemiFaceAlignImpl::trainCascade(std::map<string, vector<Point2f>>& landmarks, string path_prefix, CascadeClassifier& cascade)
{
    vector<trainingSample> samples;
    vector< vector<Point2f> > pixelCoordinates;
    fillData(samples,std::map<string, vector<Point2f>>& landmarks, string path_prefix);
    generateTestCoordinates(pixelCoordinates);
    vector< vector<regressionTree> > forest;
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        forest.push_back(gradientBoosting(samples, pixelCoordinates[i]));
        cout<<"Fitted "<<i<<"th regressor"<<endl;
    }
    return true;
}


bool KazemiFaceAlignImpl::fillData(vector<trainSample>& samples,std::map<string, vector<Point2f>>& landmarks,
                                    string path_prefix, CascadeClassifier& cascade)
{
    unsigned long currentCount =0;
    for (unsigned long i = 0; i < oversamplingAmount; ++i)
    {
        for (map<string, vector<Point2f>>::iterator dbIterator = landmarks.begin();
            dbIterator != landmarks.end(); ++dbIterator)
        {
            //Assuming the current Shape of each sample to be mean shape
            samples[currentCount].img = getImage(dbIterator->first,path_prefix);
            samples[currentCount].rect = faceDetector(samples[currentCount].img, cascade);
            extractPixelValues(samples[currentCount], dbIterator->second);
            samples[currentCount].targetShape = getRelativeShape(samples[currentCount],dbIterator->second);
            //samples[currentCount].targetShape = dbIterator->second;
            samples[currentCount].currentShape = meanShape;
            samples[currentCount].residualShape  = calcDiff(samples[currentCount].currentShape, samples[currentCount].targetShape);
        }
        ++currentCount;
    }
    cout<<currentCount<<": Training Samples Loaded.."<<endl;
    return true;
}

bool KazemiFaceAlignImpl::generateTestCoordinates(vector< vector<Point2f> >& pixelCoordinates)
{
    for (unsigned long i = 0; i < cascade_depth; ++i)
    {
        vector<Point2f> testCoordinates;
        RNG rng(time(0));
        for (unsigned long j = 0; j < num_test_coordinates; ++j)
        {
            testCoordinates.push_back(Point2f((float)rng.uniform(meanShapeBounds[0].x, meanShapeBounds[1].x), (float)rng.uniform(meanShapeBounds[0].y,meanShapeBounds[1].y)));
        }
        pixelCoordinates.push_back(testCoordinates);
    }
    return true;
}

}
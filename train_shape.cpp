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
#include <bits/stdc++.h>
#include "/home/cooper/gsoc/opencv/modules/objdetect/src/train_shape.hpp"

namespace cv{

//@ Split Feature Structure
struct splitFeature
{
    unsigned long idx1;
    unsigned long idx2;
    float thresh;
};
//@ Regression Tree structure
struct regressionTree
{
    vector<splitFeature> split;
    vector< vector<Point2f> > leaves;
};
//@ Training Sample structure
struct trainSample
{
    Mat img;
    vector<Rect> rect;
    vector<Point2f> targetShape;
    vector<Point2f> currentShape;
    vector<Point2f> residualShape;
    vector<double> pixelValues;
};

class KazemiFaceAlign : public KazemiFaceAligninclude
{
    public:
        //*@Returns number of faces detected in the image */
        int getFacesNum() const {return numFaces;}
        // /*@Returns number of landmarks to be considered */
        int getLandmarksNum() const {return numLandmarks;}
        //@ Returns cascade Depth
        int getCascadeDepth() const {return cascadeDepth;}
        //@ Sets cascade's Depth
        void setCascadeDepth(unsigned int);
        //@ Returns Tree Depth
        int getTreeDepth() const {return treeDepth;}
        //@ Sets Regression Tree's Depth
        void setTreeDepth(unsigned int);
        
        //@ Returns the left of child the Regression Tree
        unsigned long leftChild (unsigned long idx);
        //@ Returns the right child of the Regression Tree
        unsigned long rightChild (unsigned long idx);
        /*@reads the file list(xml) created by imagelist_creator.cpp */
        bool readAnnotationList(vector<cv::String>& l, string annotation_path_prefix);
        /*@Parse the txt file to extract image and its annotations*/
        bool readtxt(vector<cv::String>& filepath, std::map<string, vector<Point2f>>& landmarks, string path_prefix);
        //@ Extracts Mean Shape from the given dataset
        bool extractMeanShape(std::map<string, vector<Point2f>>& landmarks, string path_prefix,CascadeClassifier& cascade);
        //@ Applies Haar based facedetectorl
        vector<Rect> faceDetector(Mat image,CascadeClassifier& cascade);
        //@ return an image
        Mat getImage(string imgpath,string path_prefix);
        //@ Gives initial fiducial Points respective to the mean shape
        bool getInitialShape(Mat& image, CascadeClassifier& cascade);
        //@ Reads MeanShape into a vector
        bool readMeanShape();
        //@ Calculate distance between given pixel co-ordinates
        double getDistance(Point2f first , Point2f second);

        KazemiFaceAlign()
        {
            readMeanShape();
            numFaces = 1;
            numLandmarks = 194;
            cascadeDepth = 10;
            treeDepth = 4;
            num_trees_per_cascade = 500;
            nu = 0.1;
            oversamplingAmount = 20;
            feature_pool_size = 400;
            lambda = 0.1;
            numTestSplits = 20;
            numFeature = 400;
        }

    protected:
        //@ Randomly Generates splits given a set of pixel co-ordinates
        splitFeature randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates);
        //@
        splitFeature splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long begin , unsigned long end);
        //@
        bool extractPixelValues(trainSample &sample , vector<Point2f> pixelCoordinates);
        //@
        regressionTree buildRegressionTree(vector<trainSample>& samples, vector<Point2f> pixelCoordinates);
        //@
        unsigned long partitionSamples(splitFeature split, vector<trainSample>& samples, unsigned long start, unsigned long end);
    
    private:
        int numFaces;
        int numLandmarks;
        vector<Point2f> meanShape;
        vector< vector<Point2f> > initialShape;
        unsigned int cascadeDepth;
        unsigned int treeDepth;
        unsigned int num_trees_per_cascade;
        float nu;
        unsigned long oversamplingAmount;
        unsigned int feature_pool_size;
        float lambda;
        unsigned int numTestSplits;
        int numFeature;



};






void KazemiFaceAlign::setCascadeDepth(unsigned int newdepth)
{
    if(newdepth < 0)
    {
        String errmsg = "Invalid Cascade Depth";
        CV_Error(Error::StsBadArg, errmsg);
        return ;
    }
    cascadeDepth = newdepth;
}

void KazemiFaceAlign::setTreeDepth(unsigned int newdepth)
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
bool KazemiFaceAlign::readAnnotationList(vector<cv::String>& l, string annotation_path_prefix )
{
    string annotationPath = annotation_path_prefix + "*.txt";
    glob(annotationPath,l,false);
    return true;
}

//read txt files iteratively opening image and its annotations
bool KazemiFaceAlign::readtxt(vector<cv::String>& filepath, std::map<string, vector<Point2f>>& landmarks, string path_prefix)
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

bool KazemiFaceAlign::readMeanShape()
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

vector<Rect> KazemiFaceAlign::faceDetector(Mat image,CascadeClassifier& cascade)
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

Mat KazemiFaceAlign::getImage(string imgpath, string path_prefix)
{
    return imread(path_prefix + imgpath + ".jpg");
}

bool KazemiFaceAlign::extractMeanShape(std::map<string, vector<Point2f>>& landmarks, string path_prefix,CascadeClassifier& cascade)
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

bool KazemiFaceAlign::getInitialShape(Mat& image, CascadeClassifier& cascade)
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

double KazemiFaceAlign::getDistance(Point2f first , Point2f second)
{
    return sqrt(pow((first.x - second.x),2) + pow((first.y - second.y),2));
}

splitFeature KazemiFaceAlign::randomSplitFeatureGenerator(vector<Point2f>& pixelCoordinates)
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

splitFeature KazemiFaceAlign::splitGenerator(vector<trainSample>& samples, vector<Point2f> pixelCoordinates, unsigned long begin , unsigned long end )
{
    vector<splitFeature> features;
    for (unsigned int i = 0; i < numTestSplits; ++i)
        features.push_back(randomSplitFeatureGenerator(pixelCoordinates));
    vector<Mat> leftSums;
}

bool KazemiFaceAlign::extractPixelValues(trainSample &sample , vector<Point2f> pixelCoordinates)
{
    Mat image = sample.img;
    for (unsigned int i = 0; i < pixelCoordinates.size(); ++i)
    {
        cvtColor(image,image,COLOR_BGR2GRAY);
        sample.pixelValues.push_back(image.at<double>(pixelCoordinates[i]));
    }
    return true;
}
/*
KazemiFaceAlign::regressionTree KazemiFaceAlign::buildRegressionTree(vector<trainSample>& samples, vector<Point2f> pixelCoordinates)
{
    regressionTree tree;
    //Parts queue will store the extent of leaf nodes
    deque< pair<unsigned long, unsigned long > >  parts;
    parts.push_back(make_pair(0, (unsigned long)samples.size()));
    const unsigned long numSplitNodes = (unsigned long)(pow(2 , (double)getTreeDepth()) - 1);
    vector<Point2f> sums(numSplitNodes*2 + 1);
    ///----------SCOPE OF THREADING---------------------////
    for (unsigned long i = 0; i < samples.size(); i++)
    {
        //std::transform (samples[i].targetShape.begin(), samples[i].targetShape.end(), samples[i].currentShape.begin(), samples[i].residualShape.begin(), std::minus<Point2f>());
        //samples[i].residualShape = samples[i].targetShape - samples[i].currentShape;
        //sums[0] += samples[i].residualShape;
        //std::transform(samples[i].residualShape.begin(), samples[i].residualShape.end(), sums[0].begin(), sums[0].begin(), std::plus<Point2f>());
    }
    //Iteratively generate Splits in the samples
    for (unsigned long int i = 0; i < numSplitNodes; i++)
    {
        pair<unsigned long, unsigned long> rangeleaf = parts.front();
        parts.pop_front();
        splitFeature split = splitGenerator(samples, pixelCoordinates, rangeleaf.first, rangeleaf.second, sums[leftChild(i)], sums[rightChild(i)]);
        tree.split.push_back(split);
        const unsigned long mid = partitionSamples(split, samples, rangeleaf.first, rangeleaf.second);
        parts.push_back(make_pair(rangeleaf.first, mid));
        parts.push_back(make_pair(mid, rangeleaf.second));
    }
    tree.leaves.resize(parts.size());
    vector<double> currentCount[samples[0].targetShape.size()];
    //Use parts value to calculate average value of leafs
    for (unsigned long int i = 0; i < parts.size(); ++i)
    {
        //currentCount = 0;
        for (unsigned long j = parts[i].first; j < parts[i].second; ++j)
            //std::transform(samples[i].currentShape.begin(), samples[i].currentShape.end(), sums[0].begin(), sums[0].begin(), std::plus<Point2f>());
            //currentCount += samples[j].currentShape;
    }
return tree;
}
*/
unsigned long KazemiFaceAlign::partitionSamples(splitFeature split, vector<trainSample>& samples, unsigned long start, unsigned long end)
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
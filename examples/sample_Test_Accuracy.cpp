#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../include/train_shape.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

#define numSamples 100

static void help()
{
    cout << "To be written near code completion"<<endl;
}

int main(int argc, const char** argv)
{
    string cascadeName, inputName;
    CascadeClassifier cascade;
    string poseTree;
    cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade | ../data/haarcascade_frontalface_alt2.xml|}"
            "{path | ../data/300wcropped/ | }"
            "{poseTree| ../data/68_landmarks_face_align.dat |}"
            "{@filename||}"
        );
    if(parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    string path_prefix = parser.get<string>("path");
    poseTree = parser.get<string>("poseTree");
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    ifstream fs(poseTree, ios::binary);
    if(!fs.is_open())
    {
        cerr << "Failed to open trained model file " << poseTree << endl;
        help();
        return 1;
    }
    KazemiFaceAlignImpl accuracy;
    vector< vector<regressionTree> > forests;
    vector<cv::String> names;
    std::unordered_map<string, vector<Point2f>> landmarks;
    vector< vector<Point2f> > pixelCoordinates;
    accuracy.loadTrainedModel(fs, forests, pixelCoordinates);
    cout<<"Model Loaded"<<endl;
    accuracy.readnewdataset(names, landmarks, path_prefix);
    double total_error = 0;
    int count = 0;
    for (unordered_map<string, vector<Point2f> >::iterator it = landmarks.begin(); it != landmarks.end(); ++it)
    {
        if(count > numSamples)
            break;
        trainSample sample;
        sample.img = imread(it->first);
        sample.targetShape = it->second;
        accuracy.scaleData(sample.targetShape, sample.img, Size(460,460));
        sample.rect = accuracy.faceDetector(sample.img, cascade);
        vector< vector<Point2f> > resultLandmarks;
        double error_current = 0;
        if(sample.rect.size() == 1)
        {
            vector< vector<Point2f> > result;
            result = accuracy.getFacialLandmarks(sample.img, sample.rect, forests, pixelCoordinates);
            sample.currentShape = result[0];
            Mat unorm_tform  = accuracy.unnormalizing_tform(sample.rect[0]);
            for (int j = 0; j < sample.currentShape.size(); ++j)
            {
                Mat temp = (Mat_<double>(3,1)<< sample.currentShape[j].x , sample.currentShape[j].y , 1);
                Mat res = unorm_tform * temp;
                sample.currentShape[j].x = (float)(res.at<double>(0,0));
                sample.currentShape[j].y = (float)(res.at<double>(1,0));
            }
            for(unsigned long j = 0; j < sample.currentShape.size(); j++)
            {
                error_current += accuracy.getDistance(sample.targetShape[j], sample.currentShape[j]);
            }
            error_current += error_current / (sample.currentShape.size() * accuracy.getInterocularDistance(sample.currentShape));
            cout<<error_current<<endl;
            count++;
        }
        else
            continue;
        total_error += error_current;
    }
    cout<<"Total Error "<<total_error/count<<endl;
    return 0;
}
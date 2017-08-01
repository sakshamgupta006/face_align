#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "train_shape.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

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
            "{cascade | ../../opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"  //Add LBP , HOG and HAAR based detectors also
            "{path | ../data/test/ | }"
            "{poseTree| 194_landmarks_face_align.dat |}"
            "{@filename| ../data/train/213033657_1.jpg |}"
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
    inputName = parser.get<string>("@filename");  // Add multiple file support
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    KazemiFaceAlignImpl predict;
    ifstream fs(poseTree, ios::binary);
    if(!fs.is_open())
    {
        cerr << "Failed to open trained model file " << poseTree << endl;
        help();
        return 1;
    }
    vector< vector<regressionTree> > forests;
    vector<cv::String> names;
    std::unordered_map<string, vector<Point2f>> landmarks;
    vector< vector<Point2f> > pixelCoordinates;
    predict.loadTrainedModel(fs, forests, pixelCoordinates);
    predict.calcMeanShapeBounds();
    cout<<"Model Loaded"<<endl;
    predict.readAnnotationList(names, path_prefix);
    predict.readtxt(names, landmarks, path_prefix);
    double total_error = 0;
    int count = 1 ;
    for (unordered_map<string, vector<Point2f> >::iterator it = landmarks.begin(); it != landmarks.end(); ++it)
    {
        cout<<"Finding on "<<count<<endl;
        trainSample sample;
        sample.img = predict.getImage((*it).first, path_prefix);
        vector<Rect> faces = predict.faceDetector(sample.img, cascade);
        vector< vector<Point2f> > resultLandmarks;
        double error_current = 0;
        if(faces.size() == 1)
        {   
            sample.rect = faces;
            sample.targetShape = (*it).second;
            sample.currentShape = predict.getFacialLandmarks(sample, forests, pixelCoordinates);
            for(unsigned long j = 0; j < sample.currentShape.size(); j++)
            {
                error_current += predict.getDistance(sample.targetShape[j], sample.currentShape[j]);
            }
            total_error += error_current / sample.currentShape.size();
            count++;
        }
    }
    cout<<"Total Error "<<total_error/landmarks.size()<<endl;
    return 0;
}
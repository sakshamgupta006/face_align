#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../include/train_shape.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the training of Face Alignment Technique by Vahid Kazemi.\nThis approach works with LBP, HOG and HAAR based face detectors.\n"
            "This module can work on any number of landmarks on face. With some modififcation it can be used for Landmark detection of any shape\n"
            "Usage:\n"
            "./Train_ex [-cascade=<face detector model>(optional)this is the cascade module used for face detection]\n"
               "   [-path=<path to dataset> specifies the path to dataset on which the model will be trained]]\n"
               "   [-landmarks=<Number of Landmarks in the Dataset>(optional)The iBug 300w has 68 landmarks by default]\n"
               "   [-oversampling=<Number of different Initializations of each Image>(optional)(20 default)]\n"
               "   [-learningrate=<Learning Rate during Regression>(optional)(0.1 by default)]\n"
               "   [-cascadedepth=<Number of Cascade's>(optional)(10 by default)]\n"
               "   [-treedepth=<Depth of each Regression Tree>(optional)(5 by default)]\n"
               "   [-treespercascade=<Number of Tree's in each cascade>(optional)(500 by default)]\n"
               "   [-testcoordinates=<Number of test coordinates>(optional)(500 by default)]\n"
               "   [-lambda=<Priori for Randonm Feature Selection>(optional)(0.1 by default)]\n"
               "   [-samples=<Number of images to be trained on from dataset>(optional)(300 by default)]"
               "   [@filename(Output Model Filename)(\"68_landmarks_face_align.dat\" by default)]\n\n"
            "for one call:\n"
            "./Train_ex -cascade=\"../data/haarcascade_frontalface_alt2.xml\" -path=\"../data/dataset/\" -landmarks=68 -oversampling=20 -learningrate=0.1 -cascadedepth=10 -treedepth=5 -treespercascade=500 -testcoordinates=500 -lambda=0.1 -samples=300 68_landmarks_face_align.dat\n"
            "Using OpenCV version " << CV_VERSION << "\n" << endl;
}

int main(int argc, const char** argv )
{
        string cascadeName,outputName;
        CascadeClassifier cascade;
        vector<cv::String> names;
        std::unordered_map<string, vector<Point2f>> landmarks;
        string path_prefix;
        cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade | ../data/haarcascade_frontalface_alt2.xml|}"
            "{path | ../data/300wcropped/ | }"
            "{landmarks | 68 | }"
            "{oversampling | 20 |}"
            "{learningrate | 0.1 | }"
            "{cascadedepth | 10 | }"
            "{treedepth | 5 | }"
            "{treespercascade | 500 | }"
            "{testcoordinates | 400 |}"
            "{numtestsplits | 20 |}"
            "{lambda | 0.1 |}"
            "{samples | 300 |}"
            "{dataset | 1 |}"
            "{@filename| 68_landmarks_face_align.dat |}"
        );
        if(parser.has("help"))
        {
            help();
            return 0;
        }
        cascadeName = parser.get<string>("cascade");
        if(!cascade.load( cascadeName ) )
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
            help();
            return -1;
        }
        path_prefix = parser.get<string>("path");
        KazemiFaceAlignImpl train;
        unsigned long numlandmarks = parser.get<unsigned long>("landmarks");
        train.setnumLandmarks(numlandmarks);
        
        unsigned long oversampling = parser.get<unsigned long>("oversampling");
        train.setOverSampling(oversampling);
        
        float learningrate = parser.get<float>("learningrate");
        train.setLearningRate(learningrate);

        unsigned long cascadedepth = parser.get<unsigned long>("cascadedepth");
        train.setCascadeDepth(cascadedepth);

        unsigned long treedepth = parser.get<unsigned long>("treedepth");
        train.setTreeDepth(treedepth);

        unsigned long treespercascade  = parser.get<unsigned long>("treespercascade");
        train.setTreesPerCascade(treespercascade);

        unsigned long testcoordinates = parser.get<unsigned long>("testcoordinates");
        train.setTestCoordinates(testcoordinates);

        unsigned long numtestsplits = parser.get<unsigned long>("numtestsplits");
        train.setTestSplits(numtestsplits);

        float lambda = parser.get<float>("lambda");
        train.setLambda(lambda);

        unsigned long numsamples = parser.get<unsigned long>("samples");
        train.setnumSamples(numsamples);

        unsigned long dataset = parser.get<unsigned long>("dataset");
        //If dataset = 1 then it reads the 300w cropped Dataset and for 2 it reads the mirror dataset

        outputName = parser.get<string>("@filename");
        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }
        if(dataset == 1)
            train.readnewdataset(names, landmarks, path_prefix);
        else
            train.readmirror(names, landmarks, path_prefix);
        train.trainCascade(landmarks, cascade, outputName);
        cout<<"Training Complete"<<endl;
return 0;
}
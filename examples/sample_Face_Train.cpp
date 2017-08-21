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
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
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
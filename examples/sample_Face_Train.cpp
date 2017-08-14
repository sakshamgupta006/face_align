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
    cout << "To be written near code completion"<<endl;
}

int main(int argc, const char** argv )
{
        string cascadeName,outputName;
        CascadeClassifier cascade;
        bool calcmeanshape = false;
        vector<cv::String> names;
        std::unordered_map<string, vector<Point2f>> landmarks;
        string path_prefix;
        cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade | ../data/haarcascade_frontalface_alt2.xml|}"
            "{path | ../data/300wcropped/ | }"
            "{@filename| 68_landmarks_face_align.dat |}"
        );
        if(parser.has("help"))
        {
            help();
            return 0;
        }
        cascadeName= parser.get<string>("cascade");
        if(!cascade.load( cascadeName ) )
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
            help();
            return -1;
        }
        path_prefix = parser.get<string>("path");
        outputName = parser.get<string>("@filename");
        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }
        KazemiFaceAlignImpl train;
        train.readnewdataset(names, landmarks, path_prefix);
        train.trainCascade(landmarks, path_prefix, cascade, outputName);
        cout<<"Training Complete"<<endl;
return 0;
}
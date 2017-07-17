#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "train_shape.hpp"
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
        std::map<string, vector<Point2f>> landmarks;
        string path_prefix;
        cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade | ../../../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
            "{path | ../data/train/ | }"
            "{meanshape | |}"
            "{@filename| 194_landmarks_face_align.dat |}"
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
        if(parser.has("meanshape"))
        {
            calcmeanshape = true;
        }
        outputName = parser.get<string>("@filename");
        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }
        KazemiFaceAlignImpl train;
        train.readAnnotationList(names, path_prefix);
        train.readtxt(names, landmarks,path_prefix);
        if(calcmeanshape)
        {
            train.extractMeanShape(landmarks, path_prefix,cascade);
        }
        train.readMeanShape();
        train.trainCascade(landmarks, path_prefix, cascade, outputName);
        cout<<"Training Complete"<<endl;
return 0;
}
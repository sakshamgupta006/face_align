#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
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
            "{cascade | ../../../opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"  //Add LBP , HOG and HAAR based detectors also
            "{path | ../data/300wcropped/ | }"
            "{poseTree| 68_2000_100_landmarks_face_align.dat |}"
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
    ifstream fs(poseTree, ios::binary);
    if(!fs.is_open())
    {
        cerr << "Failed to open trained model file " << poseTree << endl;
        help();
        return 1;
    }
    KazemiFaceAlignImpl video;
    vector< vector<regressionTree> > forests;
    vector<cv::String> names;
    std::unordered_map<string, vector<Point2f>> landmarks;
    vector< vector<Point2f> > pixelCoordinates;
    video.loadTrainedModel(fs, forests, pixelCoordinates);
    cout<<"Model Loaded"<<endl;
    VideoCapture cap(0);
    Mat frame;
    vector< vector<Point2f>> result;  
    while(1)
    {
    	cap >> frame;
    	result = video.getFacialLandmarks(frame, forests, pixelCoordinates, cascade);
     //    vector<Point2f> currentShape = result[0];
     //    Mat unorm_tform  = video.unnormalizing_tform(sample.rect[0]);
     //    for (int j = 0; j < sample.currentShape.size(); ++j)
     //    {
     //        Mat temp = (Mat_<double>(3,1)<< sample.currentShape[j].x , sample.currentShape[j].y , 1);
     //        Mat res = unorm_tform * temp;
     //        sample.currentShape[j].x = (float)(res.at<double>(0,0));
     //        sample.currentShape[j].y = (float)(res.at<double>(1,0));
     //    }
	    // }
	    char c = (char)waitKey(10);
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
	}
    return 0;
}

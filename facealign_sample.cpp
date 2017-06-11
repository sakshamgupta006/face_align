#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <bits/stdc++.h>
#include "/home/cooper/gsoc/opencv/modules/objdetect/src/train_shape.hpp"

using namespace std;
using namespace cv;

string cascadeName;
static void help()
{
    cout << "To be written near code completion"<<endl;
}

int main(int argc, const char** argv )
{
        CascadeClassifier cascade;
        string path_prefix;
        cv::CommandLineParser parser(argc ,argv,
            "{help h||}"
            "{cascade|/home/cooper/gsoc/opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"
        );
        cascadeName= parser.get<string>("cascade");
        if( !cascade.load( cascadeName ) )
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
            //help();
            return -1;
        }
        vector<cv::String> names;
        std::map<string, vector<Point2f>> landmarks;
        path_prefix = "/home/cooper/gsoc/opencv/modules/objdetect/src/data/train/";    // need to be passed as arguments
        KazemiFaceAlign train;
        train.readAnnotationList(names, path_prefix);
        train.readtxt(names, landmarks,path_prefix);
        train.readMeanShape();
        Mat image = imread("/home/cooper/gsoc/opencv/modules/objdetect/src/data/train/14403172_1.jpg");
        imshow("initial image",image);
        train.getInitialShape(image, cascade);
        //train.extractMeanShape(landmarks, path_prefix,cascade);
return 0;
}
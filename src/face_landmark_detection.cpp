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
#include "../include/train_shape.hpp"
#include <vector>

using namespace std;
using namespace cv;

namespace cv
{

bool KazemiFaceAlignImpl::loadTrainedModel(ifstream& fs, vector< vector<regressionTree> >& forest, vector< vector<Point2f> >& pixelCoordinates)
{
    //START READING MEAN SHAPE
    size_t stringLength;
    fs.read((char*)&stringLength, sizeof(size_t));
    string meanShapestring;
    meanShapestring.resize(stringLength);
    fs.read(&meanShapestring[0], stringLength);
    if(meanShapestring != "Mean_Shape")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    unsigned long meanShapesize;
    fs.read((char*)&meanShapesize, sizeof(meanShapesize));
    meanShape.resize(meanShapesize);
    fs.read((char*)&meanShape[0], sizeof(Point2f)*meanShapesize);
    //END MEANSHAPE READING
    //START READING PIXEL COORDINATES
    fs.read((char*)&stringLength, sizeof(size_t));
    string pixelCoordinatesstring;
    pixelCoordinatesstring.resize(stringLength);
    fs.read(&pixelCoordinatesstring[0], stringLength);
    if(pixelCoordinatesstring != "Pixel_Coordinates")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    unsigned long pixelCoordinatesMainsize;
    fs.read((char*)&pixelCoordinatesMainsize, sizeof(pixelCoordinatesMainsize));
    pixelCoordinates.resize(pixelCoordinatesMainsize);
    for (unsigned long i = 0; i < pixelCoordinatesMainsize; ++i)
    {
        unsigned long pixelCoordinatesVecsize;
        fs.read((char*)&pixelCoordinatesVecsize, sizeof(pixelCoordinatesVecsize));
        pixelCoordinates[i].resize(pixelCoordinatesVecsize);
        fs.read((char*)&pixelCoordinates[i][0], sizeof(Point2f) * pixelCoordinatesVecsize);
    }
    //END PIXEL COORDINATES READING
    //START READING THE FOREST
    fs.read((char*)&stringLength, sizeof(size_t));
    string cascadedepthstring;
    cascadedepthstring.resize(stringLength);
    fs.read(&cascadedepthstring[0], stringLength);
    if(cascadedepthstring != "Cascade_Depth")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    fs.read((char*)&cascadeDepth, sizeof(cascadeDepth));
    fs.read((char*)&stringLength, sizeof(size_t));
    string numtresspercascadestring;
    numtresspercascadestring.resize(stringLength);
    fs.read(&numtresspercascadestring[0], stringLength);
    if(numtresspercascadestring != "Num_Trees_per_Cascade")
    {
        String errmsg = "Model Not Saved Properly";
        CV_Error(Error::StsBadArg, errmsg);
        return false;
    }
    fs.read((char*)&numTreesperCascade, sizeof(numTreesperCascade));
    forest.resize(cascadeDepth);
    for (unsigned long i = 0; i < cascadeDepth; ++i)
    {
        forest[i].resize(numTreesperCascade);
        for (unsigned long j = 0; j < numTreesperCascade; ++j)
        {
            unsigned long splitsize;
            fs.read((char*)&splitsize, sizeof(splitsize));
            forest[i][j].split.resize(splitsize);
            for (unsigned long k = 0; k < splitsize; ++k)
            {
                fs.read((char*)&stringLength, sizeof(size_t));
                string splitstring;
                splitstring.resize(stringLength);
                fs.read(&splitstring[0], stringLength);
                if(splitstring != "Split_Feature")
                {
                    String errmsg = "Model Not Saved Properly";
                    CV_Error(Error::StsBadArg, errmsg);
                    return false;
                }
                splitFeature temp;
                fs.read((char*)&temp, sizeof(splitFeature));
                forest[i][j].split[k] = temp;
            }
            unsigned long leavessize;
            fs.read((char*)&leavessize, sizeof(leavessize));
            forest[i][j].leaves.resize(leavessize);
            for (unsigned long k = 0; k < leavessize; ++k)
            {
                fs.read((char*)&stringLength, sizeof(size_t));
                string leafstring;
                leafstring.resize(stringLength);
                fs.read(&leafstring[0], stringLength);
                if(leafstring != "Leaf")
                {
                    String errmsg = "Model Not Saved Properly";
                    CV_Error(Error::StsBadArg, errmsg);
                    return false;
                }
                unsigned long leafsize;
                fs.read((char*)&leafsize, sizeof(leafsize));
                forest[i][j].leaves[k].resize(leafsize);
                fs.read((char*)&forest[i][j].leaves[k][0], sizeof(Point2f) * leafsize);
            }
        }
    }
//FOREST READING END
    return true;
}

vector< vector<Point2f> > KazemiFaceAlignImpl::getFacialLandmarks(Mat image, vector<Rect>& faces, vector< vector<regressionTree> >& cascadeFinal, vector< vector<Point2f>>& pixelCoordinates)
{
    float scalex = (float)image.cols / 460;
    float scaley = (float)image.rows / 460;
    Mat image2 = image.clone();
    vector< vector<Point2f> > resultPoints;
    for(unsigned long j = 0; j < faces.size(); j++)
    {
        trainSample sample;
        sample.img = image;
        sample.rect.resize(1);
        sample.rect[0] = faces[j];
        if(sample.rect.size() == 0)
            return resultPoints;
        sample.currentShape.resize(meanShape.size());
        sample.currentShape = meanShape;    
        for (int i = 0; i < cascadeFinal.size() ; ++i)
        {
            sample.pixelCoordinates = pixelCoordinates[i];
            calcRelativePixels(sample);
            extractPixelValues(sample);
            for(unsigned long j = 0; j < cascadeFinal[i].size(); j++)
            {
                unsigned long k =0 ;
                while(k < cascadeFinal[i][j].split.size())
                {
                    if ((float)sample.pixelValues[cascadeFinal[i][j].split[k].idx1] - (float)sample.pixelValues[cascadeFinal[i][j].split[k].idx2] > cascadeFinal[i][j].split[k].thresh)
                        k = leftChild(k);
                    else
                        k = rightChild(k);
                }
                k = k - cascadeFinal[i][j].split.size();
                vector<Point2f> temp;
                temp.resize(sample.currentShape.size());
                for (unsigned long l = 0; l < sample.currentShape.size(); ++l)
                {
                    temp[l] = cascadeFinal[i][j].leaves[k][l];
                }
                calcSum(sample.currentShape, temp, sample.currentShape);
            }
        }
        Mat unormmat = unnormalizing_tform(sample.rect[0]);
        for (unsigned long k = 0; k < sample.currentShape.size(); ++k)
        {
            Mat temp = (Mat_<double>(3,1)<< sample.currentShape[k].x, sample.currentShape[k].y, 1);
            Mat res = unormmat * temp;
            sample.currentShape[k].x = res.at<double>(0,0);
            sample.currentShape[k].y = res.at<double>(1,0);
        }
        resultPoints.push_back(sample.currentShape);
    }
    return resultPoints;
}


double KazemiFaceAlignImpl::getInterocularDistance (vector<Point2f>& currentShape)
{
    Point2f leftEye, rightEye;
    double count = 0;
    //Estimate the Mean of points denoting the left eye
    for (unsigned long i = 36; i <= 41; ++i) 
    {
        leftEye.x += currentShape[i].x;
        leftEye.y += currentShape[i].y;
        count++;
    }
    leftEye.x /= count;
    leftEye.y /= count;
    //Estimate the Mean of points denoting the right eye
    count = 0;
    for (unsigned long i = 42; i <= 47; ++i) 
    {
        rightEye.x += currentShape[i].x;
        rightEye.y += currentShape[i].y;
        count++;
    }
    rightEye.x /= count;
    rightEye.y /= count;

    return getDistance(leftEye, rightEye);
}

void KazemiFaceAlignImpl::display(Mat& image, vector< vector<Point2f>>& resultPoints)
{
    cout<<"Image"<<image.rows<<"  "<<image.cols<<endl;
    for(unsigned long i = 0; i < resultPoints.size(); i++)
    {
        for (unsigned long j = 0; j < resultPoints[i].size(); j++)
        {
            circle(image, Point(resultPoints[i][j]), 5, Scalar(0,0,255), -1);
            cout<<resultPoints[i][j]<<endl;
        }
    }
    imshow("Facial Landmarks", image);
}

void KazemiFaceAlignImpl::rescaleData(vector< vector<Point2f>>& results, float scalex, float scaley)
{
    for (unsigned long i = 0; i < results.size(); ++i)
    {
        for (unsigned long j = 0; j < results[i].size(); ++j)
        {
            results[i][j].x *= scalex;
            results[i][j].y *= scaley;
        }
    }
}



///----------------------------------------READING-----------------------/////


// void KazemiFaceAlignImpl :: readSplit(ifstream& is, split splitFeature &vec)
// {
//     is.read((char*)&vec, sizeof(splitr));
// }
// void KazemiFaceAlignImpl :: readLeaf(ifstream& is, vector<Point2f> &leaf)
// {
//     size_t size;
//     is.read((char*)&size, sizeof(size));
//     leaf.resize(size);
//     is.read((char*)&leaf[0], leaf.size() * sizeof(Point2f));
// }
// void KazemiFaceAlignImpl :: readPixels(ifstream& is,unsigned long index, vector<vector<Point2f>>& pixelCoordinates)
// {
//     is.read((char*)&pixelCoordinates[index][0], pixelCoordinates[index].size() * sizeof(Point2f));
// }

// bool KazemiFaceAlignImpl :: load(string filename, vector< vector<regressionTree> >& forest, vector<vector<Point2f>>& pixelCoordinates)
// {
//     if(filename.empty()){
//         String error_message = "No filename found.Aborting....";
//         CV_Error(Error::StsBadArg, error_message);
//         return false;
//     }
//     ifstream f(filename.c_str(),ios::binary);
//     if(!f.is_open()){
//         String error_message = "No file with given name found.Aborting....";
//         CV_Error(Error::StsBadArg, error_message);
//         return false;
//     }
//     size_t len;
//     f.read((char*)&len, sizeof(len));
//     char* temp = new char[len+1];
//     f.read(temp, len);
//     temp[len] = '\0';
//     string s(temp);
//     delete [] temp;
//     if(s.compare("cascade_depth")!=0){
//         String error_message = "Data not saved properly.Aborting.....";
//         CV_Error(Error::StsBadArg, error_message);
//         return false;
//     }
//     //unsigned long cascade_size;
//     f.read((char*)&cascadeDepth,sizeof(cascadeDepth));
//     loaded_forests.resize(cascadeDepth);
//     f.read((char*)&len, sizeof(len));
//     temp = new char[len+1];
//     f.read(temp, len);
//     temp[len] = '\0';
//     s = string(temp);
//     delete [] temp;
//     if(s.compare("pixel_coordinates")!=0){
//         String error_message = "Data not saved properly.Aborting.....";
//         CV_Error(Error::StsBadArg, error_message);
//         return false;
//     }
//     pixelCoordinates.resize(cascadeDepth);
//     unsigned long num_pixels;
//     f.read((char*)&num_pixels,sizeof(num_pixels));
//     for(unsigned long i=0 ; i < cascade_size ; i++){
//         pixelCoordinates[i].resize(num_pixels);
//         readPixels(f,i, pixelCoordinates);
//     }
//     f.read((char*)&len, sizeof(len));
//     temp = new char[len+1];
//     f.read(temp, len);
//     temp[len] = '\0';
//     s = string(temp);
//     delete [] temp;
//     if(s.compare("mean_shape")!=0){
//         String error_message = "Data not saved properly.Aborting.....";
//         CV_Error(Error::StsBadArg, error_message);
//         return false;
//     }
//     unsigned long mean_shape_size;
//     f.read((char*)&mean_shape_size,sizeof(mean_shape_size));
//     meanShape.resize(mean_shape_size);
//     f.read((char*)&meanShape[0], meanShape.size() * sizeof(Point2f));
//     if(!calcMeanShapeBounds())
//         exit(0);
//     f.read((char*)&len, sizeof(len));
//     temp = new char[len+1];
//     f.read(temp, len);
//     temp[len] = '\0';
//     s =string(temp);
//     delete [] temp;
//     if(s.compare("num_trees")!=0){
//         String error_message = "Data not saved properly.Aborting.....";
//         CV_Error(Error::StsBadArg, error_message);
//         return false;
//     }
//     unsigned long num_trees;
//     f.read((char*)&numTreesperCascade,sizeof(numTreesperCascade));
//     for(unsigned long i=0;i<cascadeDepth;i++){
//         for(unsigned long j=0;j<numTreesperCascade;j++){
//             regtree tree;
//             f.read((char*)&len, sizeof(len));
//             char* temp2 = new char[len+1];
//             f.read(temp2, len);
//             temp2[len] = '\0';
//             s =string(temp2);
//             delete [] temp2;
//             if(s.compare("num_nodes")!=0){
//                 String error_message = "Data not saved properly.Aborting.....";
//                 CV_Error(Error::StsBadArg, error_message);
//                 return false;
//             }
//             unsigned long num_nodes;
//             f.read((char*)&num_nodes,sizeof(num_nodes));
//             tree.nodes.resize(num_nodes+1);
//             for(unsigned long k=0; k < num_nodes ; k++){
//                 f.read((char*)&len, sizeof(len));
//                 char* temp3 = new char[len+1];
//                 f.read(temp3, len);
//                 temp3[len] = '\0';
//                 s =string(temp3);
//                 delete [] temp3;
//                 tree_node node;
//                 if(s.compare("split")==0){
//                     splitr split;
//                     readSplit(f,split);
//                     node.split = split;
//                     node.leaf.clear();
//                 }
//                 else if(s.compare("leaf")==0){
//                     vector<Point2f> leaf;
//                     readLeaf(f,leaf);
//                     node.leaf = leaf;
//                 }
//                 else{
//                     String error_message = "Data not saved properly.Aborting.....";
//                     CV_Error(Error::StsBadArg, error_message);
//                     return false;
//                 }
//                 tree.nodes[k]=node;
//             }
//             loaded_forests[i].push_back(tree);
//         }
//     }
//     f.close();
//     return true;
// }












}
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <map>
#include <set>

#include "csvHelp.h"
#include "k-near.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class Object {
public:
    int area = 0;
    int label = 0;
    int x = 0;
    int y = 0;
    int height = 0;
    int width = 0;
    cv::Point centroid;
    Vec3b color = Vec3b(0, 0, 0);
    vector<cv::Point> pixels;
};


Mat getImage(string image) {
    fs::path target_path = fs::current_path();
    target_path = (target_path / "Proj03Examples") / image;
    Mat target = imread(target_path.string(), IMREAD_COLOR);
    return target;
}

//img thresholding

Mat getThreshold(const Mat& src) {

    Mat blur;
    cv::bilateralFilter(src, blur, 20, 20 * 2, 20 / 2);
    Mat thresh(blur.size(), CV_8U);
    for (int i = 0; i < blur.rows; i++) {
        for (int j = 0; j < blur.cols; j++) {

            Vec3b intensity = blur.at<Vec3b>(i, j);
            if (intensity[0] > 50 && intensity[1] > 50 && intensity[2] > 50) {
                thresh.at<uchar>(i, j) = 0;
            }
            else {
                thresh.at<uchar>(i, j) = 255;
            }
        }
    }
    return thresh;
}

// erosion function
Mat getEroded(const Mat& src) {

    int erosion_type = MORPH_RECT;
    int erosion_size = 1;
    Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    Mat eroded;
    erode(src, eroded, element);
    return eroded;
}

// dilation operation


Mat getDilated(const Mat& src) {

    int dilation_type = MORPH_RECT;
    int dilation_size = 1;
    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    Mat dilated;
    dilate(src, dilated, element);
    return dilated;
}

// morphed image

Mat getMorphed(const Mat& src) {
    Mat dilated = getDilated(src);
    Mat eroded = getEroded(dilated);
    return eroded;
}

//Comparison function for sorting Objects

bool cmp(Object& a, Object& b) {
    return a.area > b.area;
}

//Computes Hu moments features
vector<vector<double>> calculateHuMoments(Mat& src, map<int, Object> objects) {

    Mat dst = src.clone();
    vector<vector<double>> huMoments(objects.size());
    int index = 0;
    Scalar green(0, 255, 0);
    Scalar blue(255, 0, 0);

    for (auto const& obj : objects) {
        if (obj.second.area == 0) {
            continue;
        }

        vector<Point2f> pixels;
        for (Point p : obj.second.pixels) {
            pixels.push_back(Point2f(p.y, p.x));
        }

        RotatedRect box = minAreaRect(cv::Mat(pixels));
        cv::Point2f vertices[4];
        box.points(vertices);

        for (int j = 0; j < 4; ++j) {
            cv::line(dst, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 1, 8);
        }

        Point p1, p2, p3, p4, k1, k2;
        p1 = (vertices[0] + vertices[1]) / 2;
        p2 = (vertices[2] + vertices[3]) / 2;
        p3 = (vertices[1] + vertices[2]) / 2;
        p4 = (vertices[3] + vertices[0]) / 2;

        double d1 = cv::norm(p1 - p2);
        double d2 = cv::norm(p3 - p4);

        if (d1 > d2) {
            k1 = p1;
            k2 = p2;
        }
        else {
            k1 = p3;
            k2 = p4;
        }

        Moments moment = moments(pixels);
        vector<double> hu(7);
        HuMoments(moment, hu);
        for (int j = 0; j < 7; j++) {
            hu[j] = -1 * copysign(1.0, hu[j]) * log10(abs(hu[j]));
        }

        cv::putText(dst, to_string(hu[1]), obj.second.centroid, cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(118, 185, 0), 2);
        line(dst, k1, k2, Scalar(0, 255, 0), 2);
        huMoments[index++] = hu;
    }

    imshow("Features", dst);
    return huMoments;
}

bool compareByArea(const Object& a, const Object& b) {
    return a.area > b.area;
}

// number of Connected Components

Mat getConnectedComponents(const Mat& srcImg, const Mat& targetImg, map<int, Object>& regions, int maxRegions) {

    Mat labels, stats, centroids;
    Mat outputImg = Mat::zeros(targetImg.rows, targetImg.cols, CV_8UC3);
    int numRegions = connectedComponentsWithStats(srcImg, labels, stats, centroids, 4);

    vector<Object> regionObjects;

    for (int i = 1; i < numRegions; i++) {

        int regionArea = stats.at<int>(i, cv::CC_STAT_AREA);
        if (regionArea < 1000)
            continue;
        Object regionObject;
        regionObject.area = regionArea;
        regionObject.label = i;
        regionObjects.push_back(regionObject);
    }

    sort(regionObjects.begin(), regionObjects.end(), compareByArea);

    int numSelectedRegions = maxRegions < regionObjects.size() ? maxRegions : regionObjects.size();

    for (int i = 0; i < numSelectedRegions; i++) {

        regionObjects[i].color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
        regionObjects[i].x = stats.at<int>(regionObjects[i].label, cv::CC_STAT_LEFT);
        regionObjects[i].y = stats.at<int>(regionObjects[i].label, cv::CC_STAT_TOP);
        regionObjects[i].width = stats.at<int>(regionObjects[i].label, cv::CC_STAT_WIDTH);
        regionObjects[i].height = stats.at<int>(regionObjects[i].label, cv::CC_STAT_HEIGHT);
        regionObjects[i].centroid = Point(centroids.at<double>(regionObjects[i].label, 0), centroids.at<double>(regionObjects[i].label, 1));
        regions[regionObjects[i].label] = regionObjects[i];
    }

    for (int i = 0; i < outputImg.rows; i++) {
        for (int j = 0; j < outputImg.cols; j++) {

            int label = labels.at<int>(i, j);
            regions[label].pixels.push_back(Point(i, j));
            outputImg.at<Vec3b>(i, j) = regions[label].color;
        }
    }
    return outputImg;
}



vector<vector<double>> extract_features(Mat& src, int N) {

    Mat thresh = getThreshold(src);
    Mat morphed = getMorphed(thresh);
    Mat stats, centroids;
    map<int, Object> objects;
    Mat segmented = getConnectedComponents(morphed, src, objects, N);
   // imshow("Morphed", morphed);
    //imshow("Segmented", segmented);
    vector<vector<double>>  features = calculateHuMoments(src, objects);
    //imshow("CC", segmented);
    //waitKey(0);
    return features;

}
//euclidean distance

double calculateEuclideanDistance(const vector<double>& featureVector1, const vector<double>& featureVector2) {

    double sumOfSquares = 0.0;
    for (int i = 0; i < featureVector1.size(); i++) {
        double difference = featureVector1[i] - featureVector2[i];
        sumOfSquares += difference * difference;
    }

    return sqrt(sumOfSquares);
}



// classifier using euclidean distance
string classify(const vector<vector<double>>& inputFeatures) {

    string trainingFileName = "imgtrainingFeatures_data.csv";
    std::vector<std::string> trainingLabels;
    std::vector<std::vector<double>> trainingFeatures;
    read_from_file(trainingFileName, trainingLabels, trainingFeatures);

    double minDistance = std::numeric_limits<double>::infinity();
    string minLabel;
    for (int i = 0; i < trainingFeatures.size(); i++) {

        double distance = calculateEuclideanDistance(trainingFeatures[i], inputFeatures[0]);
        if (distance < minDistance) {
            minDistance = distance;
            minLabel = trainingLabels[i];
        }
    }
    return minLabel;
}



const int menu_height = 120;
const int menu_width = 300;

const int button_height = 20;
const int button_width = 300;

const string menu_options[] = {
    "Train",
    "Save",
    "Record",
    "Exit"
};


void draw_menu(Mat& image)
{
    // Draw menu background
    rectangle(image, Point(0, 0), Point(menu_width, menu_height), Scalar(255, 0, 155), -1);

    // Draw menu options
    int y = 0;
    for (const auto& option : menu_options) {
        putText(image, option, Point(12, y + 16), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 0.2);
        rectangle(image, Point(0, 0), Point(menu_width, y + 20), Scalar(0, 255, 0), 2);
        y += button_height;
    }
}

// object detection in video mode
void object_detection() {


        Mat image(menu_height, menu_width, CV_8UC3, Scalar(200, 200, 200));
    draw_menu(image);

    namedWindow("Menu", WINDOW_NORMAL);
   // setMouseCallback("Menu", on_mouse);

    imshow("Menu", image);
   // waitKey(0);

    VideoCapture* cap;
    cap = new cv::VideoCapture(1);
    cv::namedWindow("Object Detection", 1);
    cv::Mat current_frame;
    vector<vector<double>> current_features;
    string predicted_class;

    bool is_recording = false;
    bool is_playing = false;
    VideoWriter video_writer;

    for (;;) {

        predicted_class = "No objects Detected";
        *cap >> current_frame;
        current_features = extract_features(current_frame, 1);
        if (current_features[0].size() != 0)
            predicted_class = knn_classifier(current_features, 5);
        cv::putText(current_frame, predicted_class, cv::Point(100, 100), cv::FONT_HERSHEY_TRIPLEX, 1, CV_RGB(255, 255, 255), 2);

        if (is_recording) {
            int fps;
            Size frame_size;
            if (!is_playing) {
                fps = 5;
                frame_size = current_frame.size();
                video_writer.open("Detection.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size, true);
                is_playing = true;
                cout << "Recording started" << endl;
            }
            if (is_playing)
                video_writer.write(current_frame);
        }
        else
            video_writer.release();

        imshow("Object Detection", current_frame);
        char key = cv::waitKey(10);
        string filename;

        if (key == 'q')
            break;

        if (key == 's') {
            cout << "Enter a filename: ";
            getline(cin, filename);
            imwrite("output\\" + filename + ".jpg", current_frame);
            cout << filename << " saved!" << endl;
        }

        if (key == 't') {
            cout << "Enter a filename for training image: ";
            getline(cin, filename);
            imwrite("imgtrain\\" + filename + ".jpg", current_frame);
            cout << filename << " saved!" << endl;

            String label;
            cout << "Label the object: ";
            getline(cin, label);

            string file_name = "imgtrainingFeatures_data.csv";

            if (write_to_file(file_name, label, current_features[0]))
                cout << "Successful write" << endl;
        }

        if (key == 'r') {
            is_recording = !is_recording;
            is_playing = false;
        }
    }

    delete cap;
}

// Evaulated model

void evaluate_images() {
    fs::path images_dir = fs::current_path();
    images_dir /= "imgtest";
    map<string, map<string, int>> confusionMatrix;
    set<string> unique_labels;
    float total_predictions = 0;

    for (const auto& entry : fs::directory_iterator(images_dir)) {
        fs::path image_path = entry.path();
        Mat image = imread(image_path.string(), IMREAD_COLOR);
        vector<vector<double>> image_features = extract_features(image, 1);
        string predicted_label = knn_classifier(image_features, 5);

        string file_name = image_path.filename().string();
        string true_label = file_name.substr(0, file_name.find("-"));
        unique_labels.insert(true_label);
        confusionMatrix[predicted_label][true_label] += 1;
        total_predictions++;
    }

    float correct_predictions = 0;
    for (string label : unique_labels) {
        correct_predictions += confusionMatrix[label][label];
    }

    cout << "Accuracy: " << correct_predictions / total_predictions * 100 << "%" << endl;
    string output_file_name = "confusionMatrix.csv";
    write_confusion_matrix_to_file(output_file_name, unique_labels, confusionMatrix);
}


//uncomment evaluate fucntio to find the accuracy of the system.
int main() {

   object_detection();
   // evaluate_images();
    return(0);

}

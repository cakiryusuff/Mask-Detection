#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
	VideoCapture webcam(0);
	CascadeClassifier face, mouth;
	vector<Rect> face_rec,mouth_rec;
	Mat text1_masked, text2_nomask, window;
	int cam_x, cam_y;

	text1_masked = imread("yazi.png");
	text2_nomask = imread("yazi2.png");

	cam_y = webcam.get(4);
	cam_x = webcam.get(3);

	resize(text1_masked, text1_masked, Size(cam_x, 200));
	resize(text2_nomask, text2_nomask, Size(cam_x, 200));

	window = Mat::zeros(cam_y + 200, cam_x, CV_8UC3);

	face.load("haarcascade_frontalface_default.xml");
	mouth.load("haarcascade_mcs_mouth.xml");

	if (!webcam.isOpened()) {
		cout << "Webcam Not Found...";
		return -1;
	}
	while (1) {
		Mat frame, gray;
		bool error = webcam.read(frame);
		if (!error) {
			cout << "data could not be read";
			break;
		}
		frame.copyTo(gray);
		cvtColor(gray, gray, COLOR_BGR2GRAY);
		face.detectMultiScale(gray, face_rec, 1.1, 7);

		frame.copyTo(window(Rect(0,0,frame.cols, frame.rows)));
		if (face_rec.size() == 0) {
			cout << "No Face Found" << endl;
		}
		else{
			mouth.detectMultiScale(gray, mouth_rec, 1.4, 15);
			if (mouth_rec.size() == 0) {
				text1_masked.copyTo(window(Rect(0, cam_y, text1_masked.cols, text1_masked.rows)));
				cout << "Thank You For Wearing MASK..." << endl;
			}
			else {
				text2_nomask.copyTo(window(Rect(0, cam_y, text2_nomask.cols, text2_nomask.rows)));
				cout << "Please Wear Your MASK..." << endl;
			}
		}
		imshow("text1", window);
		if (waitKey(20) == 27)
			break;
	}
}
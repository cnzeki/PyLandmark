#include "Landmark_clnf.h"
#include "LandmarkDetectorFunc.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define CVV_PI   3.1415926535897932384626433832795
#define DetectorType_MTCNN 0
#define DetectorType_SEETA 1

bool PairCompare_clnf(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first < rhs.first;
}

/* Return the indices of the top N values of vector v. */
std::vector<int> Dismin(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare_clnf);

	std::vector<int> result(N);
	for (int i = 0; i < N; ++i)
		result[i] = pairs[i].second;
	return result;
}

int Landmark_clnf::init(const char* modelDir)
{
	if (m_isInited)
	{
		return 0;
	}

	//std::string main_location = "main_clnf_general.txt";
	std::string main_location = "main_clnf_wild.txt";
	landmark_model = new LandmarkDetector::CLNF();
	if (NULL == landmark_model)
	{
		return -1;
	}

	landmark_model->init(modelDir, main_location);
	if (!landmark_model->loaded_successfully)
	{
		//cout << "ERROR: Could not load the landmark detector" << endl;
		return -2;
	}

	generateEuler(rotation_hypotheses_inits);

	cv::Mat img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	cv::Rect faceBox = cv::Rect(200, 100, 150, 150);
	vector<cv::Vec3d> rotation_hypothese;
	rotation_hypothese.push_back(cv::Vec3d(0, 0, 0));
	LandmarkDetector::DetectLandmarksInImageUseInitOri(img, faceBox, *landmark_model, landmark_param, rotation_hypothese);

	initShapes(*landmark_model);
	m_isInited = true;
	return 0;
}

int Landmark_clnf::destroy()
{
	if (!m_isInited)
	{
		return 0;
	}

	if (NULL != landmark_model)
	{
		delete landmark_model;
		landmark_model = NULL;
	}

	m_isInited = false;
	return 0;
}

double  Landmark_clnf::get_5points_dis(vector<cv::Point>& landmark1, vector<cv::Point>& landmark2)
{
	double  total_dist = 0;
	for (int i = 0; i < landmark1.size(); i++)
	{
		total_dist += sqrtf((landmark1[i].x - landmark2[i].x) * (landmark1[i].x - landmark2[i].x)
			+ (landmark1[i].y - landmark2[i].y) * (landmark1[i].y - landmark2[i].y));
	}
	return total_dist;
}

void  Landmark_clnf::conv_landmark68_2_landmark5(vector<cv::Point>& landmark5, vector<cv::Point>& landmark68)
{
	landmark5.clear();
	// left eye
	cv::Point  left_eye68 = cv::Point(0, 0);
	for (int i = 36; i < 42; i++)
	{
		left_eye68.x += landmark68[i].x;
		left_eye68.y += landmark68[i].y;
	}
	left_eye68.x /= 6;
	left_eye68.y /= 6;
	landmark5.push_back(left_eye68);

	// right eye
	cv::Point  right_eye68 = cv::Point(0, 0);
	for (int i = 42; i < 48; i++)
	{
		right_eye68.x += landmark68[i].x;
		right_eye68.y += landmark68[i].y;
	}
	right_eye68.x /= 6;
	right_eye68.y /= 6;
	landmark5.push_back(right_eye68);

	landmark5.push_back(landmark68[30]);  //nose
	landmark5.push_back(landmark68[48]);  //mouth_left
	landmark5.push_back(landmark68[54]);  //mouth_right
}

void  Landmark_clnf::use_pdm_get_initShape(cv::Rect& face, LandmarkDetector::CLNF& landmark_model, cv::Vec3d& rotation_hypothese, vector<cv::Point>& landmark)
{
	landmark.clear();

	// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
	landmark_model.pdm.CalcParams(landmark_model.params_global, face, landmark_model.params_local, rotation_hypothese);

	// Placeholder for the landmarks
	cv::Mat_<float> current_shape(2 * landmark_model.pdm.NumberOfPoints(), 1, 0.0f);
	landmark_model.pdm.CalcShape2D(current_shape, landmark_model.params_local, landmark_model.params_global);
	int ld_n = current_shape.rows / 2;
	for (int i = 0; i < ld_n; ++i)
	{
		cv::Point featurePoint(cvRound(current_shape.at<float>(i)), cvRound(current_shape.at<float>(i + ld_n)));
		landmark.push_back(featurePoint);
	}
}

void  Landmark_clnf::generateEuler(vector<cv::Vec3d>& rotation_hypothese)
{
	rotation_hypothese.clear();
	float  pitch_degree = 0;
	float  yaw_degree = 0;
	float  roll_degree = 0;
	for (int pitch = -30; pitch <= 30; pitch += 10)
	{
		pitch_degree = pitch * CVV_PI / 180.0;
		for (int yaw = -90; yaw <= 90; yaw += 10)
		{
			yaw_degree = yaw * CVV_PI / 180.0;
			for (int roll = -50; roll <= 50; roll += 10)
			{
				roll_degree = roll * CVV_PI / 180.0;
				rotation_hypothese.push_back(cv::Vec3d(pitch_degree, yaw_degree, roll_degree));
			}
		}
	}
}


void  Landmark_clnf::refineModelLdmark(vector<cv::Point>& landmark5)
{
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;

	float  dist = sqrtf((landmark5[0].x - landmark5[1].x)*(landmark5[0].x - landmark5[1].x)
		+ (landmark5[0].y - landmark5[1].y)*(landmark5[0].y - landmark5[1].y));

	float  scale = dist / FACE_EYE_DIST;

	for (int i = 0; i < landmark5.size(); i++)
	{
		int off_x = landmark5[i].x - m_cx;
		int off_y = landmark5[i].y - m_cy;
		landmark5[i].x = m_cx + off_x / scale;
		landmark5[i].y = m_cy + off_y / scale;
	}
}

void  Landmark_clnf::initShapes(LandmarkDetector::CLNF& landmark_model)
{
	cv::Rect face = cv::Rect(FACE_RECT_X, FACE_RECT_Y, FACE_RECT_W, FACE_RECT_H);
	m_shapeCaches.clear();
	for (int i = 0; i < rotation_hypotheses_inits.size(); i++)
	{
		vector<cv::Point> landmark;
		use_pdm_get_initShape(face, landmark_model, rotation_hypotheses_inits[i], landmark);

		shapeCaches  shape;
		shape.rotation_hypothese = rotation_hypotheses_inits[i];
		conv_landmark68_2_landmark5(shape.landmark5, landmark);

		refineModelLdmark(shape.landmark5);
		m_shapeCaches.push_back(shape);
	}
}


void  Landmark_clnf::getCurLdmarkCof(cv::Rect& face, vector<cv::Point>& landmark5, float& offset_x, float& offset_y, float& scale)
{
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;
	int  c_cx = face.x + face.width / 2;
	int  c_cy = face.y + face.height / 2;
	offset_x = c_cx - m_cx;
	offset_y = c_cy - m_cy;

	float  dist = sqrtf((landmark5[0].x - landmark5[1].x)*(landmark5[0].x - landmark5[1].x)
		+ (landmark5[0].y - landmark5[1].y)*(landmark5[0].y - landmark5[1].y));

	scale = dist / FACE_EYE_DIST;
}


void Landmark_clnf::refineLdmark(cv::Mat& img, cv::Rect& face, vector<cv::Point>& landmark5)
{
	float offset_x, offset_y, scale;
	getCurLdmarkCof(face, landmark5, offset_x, offset_y, scale);

	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;
	int  c_cx = face.x + face.width / 2;
	int  c_cy = face.y + face.height / 2;
	for (int i = 0; i < landmark5.size(); i++)
	{
		landmark5[i].x = m_cx + (landmark5[i].x - c_cx) / scale;
		landmark5[i].y = m_cy + (landmark5[i].y - c_cy) / scale;
	}
}

void  Landmark_clnf::clc_shape_pdm(cv::Mat& img, cv::Rect& face, vector<cv::Point>& landmark5, int faceDetectType, vector<cv::Vec3d>& rotation_hypothese)
{
	rotation_hypothese.clear();

	if (faceDetectType != DetectorType_MTCNN)
	{
		rotation_hypothese.push_back(cv::Vec3d(0, 0, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, -0.5236, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 0.5236, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, -0.96, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 0.96, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 0, 0.5236));
		rotation_hypothese.push_back(cv::Vec3d(0, 0, -0.5236));
		rotation_hypothese.push_back(cv::Vec3d(0, -1.57, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, 1.57, 0));
		rotation_hypothese.push_back(cv::Vec3d(0, -1.22, 0.698));
		rotation_hypothese.push_back(cv::Vec3d(0, 1.22, -0.698));
		return;
	}

	vector<cv::Point> refine_landmark5(landmark5);
	refineLdmark(img, face, refine_landmark5);

	vector<float> dist_caches;
	int id = 0;
	float  min_dis = 100000;
	for (int i = 0; i < m_shapeCaches.size(); i++)
	{
		double dis = get_5points_dis(refine_landmark5, m_shapeCaches[i].landmark5);
		dist_caches.push_back(dis);
		if (dis < min_dis)
		{
			min_dis = dis;
			id = i;
		}
	}
	rotation_hypothese.push_back(m_shapeCaches[id].rotation_hypothese);
}
	
//// Correct the box to expectation to be tight around facial landmarks  FaceDetectMTCNN.cpp ---887
void  Landmark_clnf::refineMtcnnBox(cv::Rect& box, vector<cv::Point>& landmarks, int faceDetectType)
{
	if (faceDetectType != DetectorType_MTCNN)
	{
		return;
	}

	box.x = box.width * -0.0075 + box.x;
	box.y = box.height * 0.2459 + box.y;
	box.width = 1.0323 * box.width;
	box.height = 0.7751 * box.height;
}


void Landmark_clnf::getInitShape(cv::Mat& img, cv::Rect mtcnn_faceBox, vector<cv::Point> mtcnn_ldmark, vector<cv::Point>& init_landmark)
{
	int  faceDetectType = DetectorType_MTCNN;
	if (mtcnn_ldmark.size() == 0)
	{
		faceDetectType = DetectorType_SEETA;
	}

	cv::Rect  faceBox = mtcnn_faceBox;
	refineMtcnnBox(faceBox, mtcnn_ldmark, faceDetectType);

	vector<cv::Vec3d> rotation_hypothese;
	clc_shape_pdm(img, faceBox, mtcnn_ldmark, faceDetectType, rotation_hypothese);

	init_landmark.clear();
	use_pdm_get_initShape(faceBox, *landmark_model, rotation_hypothese[0], init_landmark);
}

int Landmark_clnf::landmark(unsigned char* yuv, int width, int height, cv::Rect rect, std::vector<cv::Point>  landmark5, std::vector<cv::Point>& ldmark_pts,int detectorType)
{
	if (!m_isInited)
	{
		return -1;
	}

	if ((width < 1) || (height < 1) || (NULL == yuv) || rect.area() < 1)
	{
		return -2;
	}
		
	int  depth = CV_8UC3;
	cv::Rect faceBox = rect;
	cv::Mat image(height, width, depth, yuv);

	refineMtcnnBox(faceBox, landmark5, detectorType);
		
	vector<cv::Vec3d> rotation_hypothese;
	clc_shape_pdm(image, faceBox, landmark5, detectorType, rotation_hypothese);
				
	bool success = LandmarkDetector::DetectLandmarksInImageUseInitOri(image, faceBox, *landmark_model, landmark_param, rotation_hypothese);
		
	ldmark_pts.clear();
	int n = landmark_model->detected_landmarks.rows / 2;
	for (int l = 0; l < n; l++) 
	{
		cv::Point featurePoint(cvRound(landmark_model->detected_landmarks.at<float>(l)), cvRound(landmark_model->detected_landmarks.at<float>(l + n)));
		ldmark_pts.push_back(featurePoint);
	}

	return 0;
}
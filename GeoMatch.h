#pragma once
#include <QMainWindow>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<set>
#include<string>
#include<math.h>


class GeoMatch {
private:

	int noOfCordinates; //�����е�Ԫ�ظ���
	cv::Point* cordinates; //���������ڴ洢ģ�͵������
	int modelHeight; //ģ��߶�
	int modelWidth; //ģ����
	double* edgeMagnitude; //�ݶ�
	double* edgeDerivativeX; //x�����ݶ�
	double* edgeDerivativeY; //y�����ݶ�
	cv::Point centerOfGravity; //ģ������
	bool modelDefined;


	void CreateDoubleMatrix(double**& matrix, cv::Size size);
	void ReleaseDoubleMatrix(double**& matrix, int size);

	
public:

	typedef struct _MatchResult {
		bool invalid;
		cv::Point bestLoc;
		double bestAngle;
		double bestValue;
	} MatchResult;

	GeoMatch();
	~GeoMatch();

	
	std::vector<MatchResult> PyramidMatching(cv::Mat& srcarr, cv::Mat& templateArr, 
		double maxContrast = 100.0, double minContrast = 66.0, double thresh = 0.73, double greediness = 0.8, int pyramidLayers = 3);
    cv::Mat DrawMatch(cv::Mat pImage, cv::Size templsize, std::vector<GeoMatch::MatchResult> matchResultFilt, cv::Scalar color, int lineWidth);
    cv::Mat QImage2Mat(QImage& img);
    QImage Mat2QImage(cv::Mat& mat);

private:
	int CreateGeoMatchModel(cv::Mat templateArr, double maxContrast, double minContrast);
	double FindGeoMatchModel(cv::Mat srcarr, double minScore, double greediness, cv::Mat* result);
	int RoughMatch(cv::Mat srcarr, cv::Mat templateArr, 
		double maxContrast, double minContrast, double thresh, double greediness, std::vector<MatchResult>& candResults);
	int MatchInLayer(cv::Mat& srcarr, cv::Mat& templateArr, cv::Point& bestLoc, double& bestAngle,
		double& bestValue, double maxContrast, double minContrast, double greediness, double thresh,
		cv::Point guessLoc = cv::Point(0, 0), double guessAngle = 0.0, double angleSearchMax = 180., double angleStep = 0.1);
};

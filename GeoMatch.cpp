#include<iostream>
#include"GeoMatch.h"

GeoMatch::GeoMatch() {
	noOfCordinates = 0; //初始化为0
	modelDefined = false;
}

int GeoMatch::RoughMatch(cv::Mat srcarr, cv::Mat templateArr, double thresh, double greediness,
	double maxContrast, double minContrast, std::vector<MatchResult>& candResults) {
	//类型检测
	if (srcarr.type() != CV_8UC1 || templateArr.type() != CV_8UC1) {
		std::cout << "srcarr's type or templateArr's type is wrong in function RoughMatch." << std::endl;
		return 0;
	}

	//初始化
	cv::Mat resultScore = cv::Mat::zeros(srcarr.size(), CV_32FC1);
	cv::Mat resultAngle = cv::Mat::zeros(srcarr.size(), CV_32FC1);
	cv::Mat resultBin = cv::Mat::zeros(srcarr.size(), CV_8UC1);
	cv::Point2f templCenter = cv::Point2f(templateArr.cols / 2, templateArr.rows / 2);

	int resultCols = 0, resultRows = 0;

	//多角度
	for (int angle = -180; angle < 180; ++angle) {
		
		//无损旋转模板
		cv::Mat rotMat = cv::getRotationMatrix2D(templCenter, angle, 1);
		cv::Rect rotBbox = cv::RotatedRect(templCenter, templateArr.size(), angle).boundingRect();
		rotMat.at<double>(0, 2) += rotBbox.width / 2.0 - templCenter.x;
		rotMat.at<double>(1, 2) += rotBbox.height / 2.0 - templCenter.y;
		cv::Mat templWarpped;
		cv::warpAffine(templateArr, templWarpped, rotMat, rotBbox.size());

		//基于边缘的模板匹配
		CreateGeoMatchModel(templWarpped, maxContrast, minContrast);
		cv::Mat result = cv::Mat::zeros(srcarr.size(), CV_32FC1);
		FindGeoMatchModel(srcarr, thresh, greediness, &result);

		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

		if (maxVal < thresh) {
			continue;
		}
		
		for (int i = 0; i < result.rows; ++i) {
			for (int j = 0; j < result.cols; ++j) {
				float value = result.at<float>(i, j);
				if (value > thresh) {
					resultBin.at<uchar>(i, j) = 255;
					float preBestValue = resultScore.at<float>(i, j);
					if (value > preBestValue) {
						resultScore.at<float>(i, j) = value;
						resultAngle.at<float>(i, j) = angle;
					}
				}
			}
		}
	}

	cv::Mat labelsImg;
    //利用连通图去重
	int nncomps = cv::connectedComponents(resultBin, labelsImg, 8, CV_32S);

	for (int i = 1; i < nncomps; i++) {
		cv::Mat maskImg = (labelsImg == i);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(resultScore, &minVal, &maxVal, &minLoc, &maxLoc, maskImg);
		
		cv::Point bestLoc = maxLoc;
		MatchResult candidateResult;

		candidateResult.invalid = false;
		candidateResult.bestAngle = resultAngle.at<float>(bestLoc);
		candidateResult.bestValue = resultScore.at<float>(bestLoc);
		candidateResult.bestLoc = bestLoc;
		candResults.push_back(candidateResult);
		cv::circle(srcarr, bestLoc, 2, cv::Scalar(255, 0, 0), 2);
	}
	return 1;
}

int GeoMatch::MatchInLayer(cv::Mat& srcarr, cv::Mat& templateArr, cv::Point& bestLoc, double& bestAngle, double& bestValue,
	double maxContrast, double minContrast, double greediness, double thresh, cv::Point guessLoc, double guessAngle, double angleSearchMax, double angleStep) {
	bestValue = -1;
	bestAngle = 0.0;
	cv::Mat imgROI;
	cv::Point imgROIOriginInSrc;
	cv::Point2f templCenter;

	int margin;
	cv::Mat resultScore = cv::Mat::zeros(srcarr.size(), CV_32FC1);
	cv::Mat resultAngle = cv::Mat::zeros(srcarr.size(), CV_32FC1);
	cv::Mat resultBin = cv::Mat::zeros(srcarr.size(), CV_8UC1);
	templCenter = cv::Point2f(templateArr.cols / 2.0, templateArr.rows / 2.0);

	margin = 5;
	cv::Rect maxBbox = cv::RotatedRect(templCenter, templateArr.size(), guessAngle).boundingRect();

	double roiDx = maxBbox.width / 2.0;
	double roiDy = maxBbox.height / 2.0;
	//边界检查，局部匹配
	//矩形左上坐标
	int roiLx = (guessLoc.x - roiDx - margin) > 0 ? (guessLoc.x - roiDx - margin) : 0;
	int roiLy = (guessLoc.y - roiDy - margin) > 0 ? (guessLoc.y - roiDy - margin) : 0;

	//局部的宽高
	int roiW = ((guessLoc.x - roiDx + maxBbox.width + 2 * margin) < srcarr.cols - 1 && roiLx > 0) ? (maxBbox.width + 2 * margin) : (srcarr.cols - roiLx);
	int roiH = ((guessLoc.y - roiDy + maxBbox.height + 2 * margin) < srcarr.rows - 1 && roiLy > 0) ? (maxBbox.height + 2 * margin) : (srcarr.rows - roiLy);

	imgROI = srcarr(cv::Rect(cv::Rect(roiLx, roiLy, roiW, roiH)));
	imgROIOriginInSrc = cv::Point(roiLx, roiLy);
	cv::Mat result;
	for (double angle = guessAngle - angleSearchMax; angle < guessAngle + angleSearchMax; angle += angleStep) {
		//无损旋转图像
		cv::Mat rotMat = cv::getRotationMatrix2D(templCenter, angle, 1.0);
		cv::Rect rotBbox = cv::RotatedRect(templCenter, templateArr.size(), angle).boundingRect();

		rotMat.at<double>(0, 2) += rotBbox.width / 2.0 - templCenter.x;
		rotMat.at<double>(1, 2) += rotBbox.height / 2.0 - templCenter.y;
		cv::Mat templWarpped;
		cv::warpAffine(templateArr, templWarpped, rotMat, rotBbox.size());

		//基于边缘的模板匹配
		CreateGeoMatchModel(templWarpped, maxContrast, minContrast);
		result = cv::Mat::zeros(srcarr.size(), CV_32FC1);
		FindGeoMatchModel(imgROI, thresh, greediness, &result);

		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

		if (bestValue < maxVal) {
			bestLoc = imgROIOriginInSrc + maxLoc;
			bestValue = maxVal;
			bestAngle = angle;
		}

	}
	return 1;
}


std::vector<GeoMatch::MatchResult> GeoMatch::PyramidMatching(cv::Mat& srcarr, cv::Mat& templateArr, 
									double maxContrast, double minContrast,
									double thresh, double greediness, int pyramidLayers) {
	std::vector<MatchResult> matchResults;

	cv::Mat srcExpand = srcarr;
	cv::Mat templExpand = templateArr;

	int bordWidth, srcBordWidth, templBordWidth;
	int srcExpandRatio = 3;
	int templExpandRatio = 1;

	bordWidth = std::pow(2, pyramidLayers - 1);
	srcBordWidth = srcExpandRatio * bordWidth;
	templBordWidth = templExpandRatio * bordWidth;

	if (pyramidLayers > 4) {
		//高斯滤波
		cv::copyMakeBorder(srcarr, srcExpand, srcBordWidth, srcBordWidth, srcBordWidth, srcBordWidth, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		cv::copyMakeBorder(templateArr, templExpand, templBordWidth, templBordWidth, templBordWidth, templBordWidth, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		int smoothRadius = 2 * bordWidth + 1;
		cv::GaussianBlur(srcExpand, srcExpand, cv::Size(smoothRadius, smoothRadius), 0);
		cv::GaussianBlur(templExpand, templExpand, cv::Size(smoothRadius, smoothRadius), 0);

		cv::normalize(srcExpand, srcExpand, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::normalize(templExpand, templExpand, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	}

	std::vector<cv::Mat> pyramidSrcImgs, pyramidTemplImgs;
	pyramidSrcImgs.push_back(srcExpand);
	pyramidTemplImgs.push_back(templExpand);
	//下采样构建金字塔
	for (int i = 1; i < pyramidLayers; ++i) {
		cv::Mat pyramidSrcImg, pyramidTemplImg;
		cv::pyrDown(pyramidSrcImgs.back(), pyramidSrcImg);
		cv::pyrDown(pyramidTemplImgs.back(), pyramidTemplImg);
		pyramidSrcImgs.push_back(pyramidSrcImg);
		pyramidTemplImgs.push_back(pyramidTemplImg);
	}

	int matchResult = 1;
	MatchResult candResult;
	for (int i = pyramidLayers; i > 0; --i) {
		double threshInLayer = thresh;
        //对金字塔顶层进行粗匹配
        if (i == pyramidLayers) {
			RoughMatch(pyramidSrcImgs.at(i - 1), pyramidTemplImgs.at(i - 1), threshInLayer, greediness, maxContrast, minContrast, matchResults);
			for (int candIndex = 0; candIndex < matchResults.size(); candIndex++) {
				candResult = matchResults[candIndex];
				candResult.bestLoc = candResult.bestLoc * std::pow(2, i - 1);
				matchResults[candIndex] = candResult;
			}
		}
		//其余层根据上一层匹配结果微调
        else {
			for (int candIndex = 0; candIndex < matchResults.size(); candIndex++) {
				cv::Point guessLoc, bestLoc;
				double bestAngle, bestValue, guessAngle;
				double angleSearchMax, angleStep;
				if (i == 1) {
					angleSearchMax = 0.5;
					angleStep = 0.1;
				}
				else {
					angleSearchMax = 2;
					angleStep = 0.2;
				}
				
				MatchResult& candResult = matchResults[candIndex];
				
				if (candResult.invalid) {
					continue;
				}

				guessLoc = candResult.bestLoc / std::pow(2, i - 1);
				guessAngle = candResult.bestAngle;
				matchResult = MatchInLayer(pyramidSrcImgs.at(i - 1), pyramidTemplImgs.at(i - 1), bestLoc, bestAngle, bestValue,
					maxContrast, minContrast, greediness, threshInLayer, guessLoc, guessAngle, angleSearchMax, angleStep);

				candResult.bestLoc = bestLoc * std::pow(2, i - 1);
				candResult.bestAngle = bestAngle;
				candResult.bestValue = bestValue;

				if (bestValue < thresh) {
					candResult.invalid = true;
				}
			}
		}
        std::cout<< "Layer "<< i <<" matching completed." << std::endl;

		if (!matchResult) break;
	}

	if (pyramidLayers > 4) {
		for (auto& result : matchResults) {
			cv::Mat newTempl = pyramidTemplImgs.at(0);
			cv::Point2f newTemplCenter = cv::Point2f(newTempl.cols / 2.0, newTempl.rows / 2.0);
			float bestAngle = result.bestAngle;

			cv::Rect rotBboxOrigin = cv::RotatedRect(newTemplCenter, templateArr.size(), bestAngle).boundingRect();
			cv::Rect rotBboxExpandBorder = cv::RotatedRect(newTemplCenter, newTempl.size(), bestAngle).boundingRect();


			result.bestLoc.x += (rotBboxExpandBorder.width - rotBboxOrigin.width) / 2.0 - srcBordWidth;
			result.bestLoc.y += (rotBboxExpandBorder.height - rotBboxOrigin.height) / 2.0 - srcBordWidth;
		}
	}
	std::vector<MatchResult> matchResultFilt;
	int size = 3;
	for (auto result : matchResults) {
		bool flag = true;
		if (!result.invalid) {
            //去重 中心点过近的就去掉
			for (int i = 0; i < matchResultFilt.size(); i++) {
				if (matchResultFilt[i].bestLoc.x - size < result.bestLoc.x &&
					matchResultFilt[i].bestLoc.x + size > result.bestLoc.x &&
					matchResultFilt[i].bestLoc.y - size < result.bestLoc.y &&
					matchResultFilt[i].bestLoc.y + size > result.bestLoc.y) {
					flag = false;
					break;
				}
			}

			if (flag) {
				matchResultFilt.push_back(result);
			}

		}
    }
    std::cout << "Match completed." << std::endl;
	return matchResultFilt;
}


int GeoMatch::CreateGeoMatchModel(cv::Mat templateArr, double maxContrast, double minContrast) {
	cv::Mat gx; //存储X梯度
	cv::Mat gy; //存储y梯度
	cv::Mat nmsEdges; //存储模板的边缘
	cv::Size Ssize;  //记录模板大小

	cv::Mat src(templateArr); //复制图像
	if (src.type() != CV_8UC1)
	{
		return 0;
	}

	//设置宽、高
	Ssize.width = src.cols;
	Ssize.height = src.rows;
	modelWidth = src.cols; //保存模板宽
	modelHeight = src.rows; //保存模板高

	noOfCordinates = 0; //初始化
	
	//分配空间
	cordinates = new cv::Point[modelWidth * modelHeight];
	edgeMagnitude = new double[modelWidth * modelHeight];
	edgeDerivativeX = new double[modelWidth * modelHeight];
	edgeDerivativeY = new double[modelWidth * modelHeight];

	gx = cv::Mat(Ssize.height, Ssize.width, CV_16SC1);
	gy = cv::Mat(Ssize.height, Ssize.width, CV_16SC1);

	cv::Sobel(src, gx, CV_16S, 1, 0, 3); //计算x方向梯度
	cv::Sobel(src, gy, CV_16S, 0, 1, 3); //计算y方向梯度

	nmsEdges = cv::Mat(Ssize.height, Ssize.width, CV_32F);

    const short* _sdx;
    const short* _sdy;
	double fdx, fdy;
	double MagG, DirG;
	double MaxGradient = -999999.99;
	double direction;
	int* orients = new int[Ssize.height * Ssize.width];

	int count = 0; //计数

	double** magMat;
	CreateDoubleMatrix(magMat, Ssize);

	for (int i = 1; i < Ssize.height - 1; ++i) {
		for (int j = 1; j < Ssize.width - 1; ++j) {
			fdx = gx.at<short>(i, j);
			fdy = gy.at<short>(i, j);

			MagG = sqrt((float)(fdx * fdx) + (float)(fdy * fdy)); //Magnitude = sqrt(fdx^2 + fdy^2)
			direction = atan2f((float)fdy, (float)fdx);
			magMat[i][j] = MagG;

			if (MagG > MaxGradient) {
				MaxGradient = MagG; //获取最大梯度用于归一化
			}

			//取最接近的角度 0,45,90,135
			if ((direction > 0 && direction < 22.5) || (direction > 157.5 && direction < 202.5) || (direction > 337.5 && direction < 360)) {
				direction = 0;
			}
			else if ((direction > 22.5 && direction < 67.5) || (direction > 202.5 && direction < 247.5)) {
				direction = 45;
			}
			else if ((direction > 67.5 && direction < 112.5) || (direction > 247.5 && direction < 292.5)) {
				direction = 90;
			}
			else if ((direction > 112.5 && direction < 157.5) || (direction > 292.5 && direction < 337.5)) {
				direction = 135;
			}
			else {
				direction = 0;
			}

			orients[count] = (int)direction;
			count++;
		}
	}

	count = 0;


	//非最大值抑制
	double leftPixel, rightPixel;

	for (int i = 1; i < Ssize.height - 1; i++) {
		for (int j = 1; j < Ssize.width - 1; j++) {
			switch (orients[count])
			{
			case 0: //水平
				leftPixel = magMat[i][j - 1];
				rightPixel = magMat[i][j + 1];
				break;
			case 45: //斜向45
				leftPixel = magMat[i - 1][j + 1];
				rightPixel = magMat[i + 1][j - 1];
				break;
			case 90: //竖直
				leftPixel = magMat[i - 1][j];
				rightPixel = magMat[i + 1][j];
			case 135: //斜向135
				leftPixel = magMat[i - 1][j - 1];
				rightPixel = magMat[i + 1][j + 1];
				break;
			}

			//沿梯度方向进行非极大值抑制
			if ((magMat[i][j] < leftPixel) || (magMat[i][j] < rightPixel)) {
				nmsEdges.at<float>(i, j) = 0.0f;
			}
			else {
				nmsEdges.at<float>(i, j) = (uchar)((magMat[i][j] / MaxGradient) * 255);
			}

			count++;
		}
	}

	//滞后阈值
	int RSum = 0, CSum = 0;
	int curX, curY;
	int flag = 1;
	for (int i = 1; i < Ssize.height - 1; ++i) {
		for (int j = 1; j < Ssize.width - 1; ++j) {
			fdx = gx.at<short>(i, j);
			fdy = gy.at<short>(i, j);

			MagG = sqrt(fdx * fdx + fdy * fdy);
			DirG = atan2f((float)fdy, (float)fdx);
			flag = 1;
			double val = nmsEdges.at<float>(i, j);
			if (val < maxContrast) {
				if (val < minContrast) {
					nmsEdges.at<float>(i, j) = 0.0;
					flag = 0;
				}
				else {
                    //寻找周围八个点有没有强边缘
					if ((nmsEdges.at<float>(i - 1, j - 1) < maxContrast) &&
						(nmsEdges.at<float>(i - 1, j) < maxContrast) &&
						(nmsEdges.at<float>(i - 1, j + 1) < maxContrast) &&
						(nmsEdges.at<float>(i, j - 1) < maxContrast) &&
						(nmsEdges.at<float>(i, j + 1) < maxContrast) &&
						(nmsEdges.at<float>(i + 1, j - 1) < maxContrast) &&
						(nmsEdges.at<float>(i + 1, j) < maxContrast) &&
						(nmsEdges.at<float>(i + 1, j + 1) < maxContrast)) {
						nmsEdges.at<float>(i, j) = 0;
						flag = 0;
					}
				}
			}
			
			curX = i;
			curY = j;
			if (flag != 0) {
				if (fdx != 0 || fdy != 0) {
					//边缘点的行和 和 列和
					RSum += curX;
					CSum += curY;

					cordinates[noOfCordinates].x = curX;
					cordinates[noOfCordinates].y = curY;
					edgeDerivativeX[noOfCordinates] = fdx;
					edgeDerivativeY[noOfCordinates] = fdy;

					//避免除0
					if (MagG != 0) {
						edgeMagnitude[noOfCordinates] = 1 / MagG;
					}
					else {
						edgeMagnitude[noOfCordinates] = 0;
					}
					noOfCordinates++;
				}
			}
		}
	}

	//边缘中心坐标
	centerOfGravity.x = RSum / noOfCordinates; 
	centerOfGravity.y = CSum / noOfCordinates; 
	
	
	for (int m = 0; m < noOfCordinates; ++m) {
		//获取边缘相对中心的坐标
		cordinates[m].x -= centerOfGravity.x;
		cordinates[m].y -= centerOfGravity.y;
	}

	delete[] orients;

	gx.release();
	gy.release();
	nmsEdges.release();

	ReleaseDoubleMatrix(magMat, Ssize.height);

	modelDefined = true;
	return 1;
}


double GeoMatch::FindGeoMatchModel(cv::Mat srcarr, double minScore, double greediness, cv::Mat* result) {
	cv::Mat Sdx, Sdy;

	double partialSum = 0.0;
	double sumOfCoords = 0.0;
	double partialScore = 0.0;
	const short* _Sdx;
	const short* _Sdy;
	double iTx, iTy, iSx, iSy;
	double gradMag;
	int curX, curY;

	double** matGradMag; //梯度矩阵
	
	cv::Mat src(srcarr);
	if (srcarr.type() != CV_8UC1 || !modelDefined) {
			return 0;
	}

	if (result->type() != CV_32FC1) {
		std::cout << "result's type is wrong." << std::endl;
		return 0;
	}

	cv::Size Ssize;
	Ssize.width = src.cols;
	Ssize.height = src.rows;

	CreateDoubleMatrix(matGradMag, Ssize); // 创建矩阵存储梯度
	Sdx = cv::Mat(Ssize.height, Ssize.width, CV_16SC1);
	Sdy = cv::Mat(Ssize.height, Ssize.width, CV_16SC1);

	cv::Sobel(src, Sdx, CV_16S, 1, 0, 3);
	cv::Sobel(src, Sdy, CV_16S, 0, 1, 3);
	
	double normMinScore = minScore / noOfCordinates;
	double normGreediness = ((1 - greediness * minScore) / (1 - greediness)) / noOfCordinates;
	
	for (int i = 0; i < Ssize.height; ++i) {
		for (int j = 0; j < Ssize.width; ++j) {
			
			iSx = Sdx.at<short>(i, j);
			iSy = Sdy.at<short>(i, j);

			gradMag = sqrt(iSx * iSx + iSy * iSy);
			if (gradMag != 0) {
				matGradMag[i][j] = 1 / gradMag;
			}
			else {
				matGradMag[i][j] = 0;
			}
		}
	}

	for (int i = 0; i < Ssize.height; ++i) {
		for (int j = 0; j < Ssize.width; ++j) {
			partialSum = 0;
			for (int m = 0; m < noOfCordinates; ++m) {
				//curX, curY 是以（i,j）为中心的边缘的坐标
				curX = i + cordinates[m].x;
				curY = j + cordinates[m].y;
				iTx = edgeDerivativeX[m];
				iTy = edgeDerivativeY[m];

				if (curX < 0 || curY <0 || curX > Ssize.height - 1 || curY>Ssize.width - 1) {
					continue; //越界情况
				}

				iSx = Sdx.at<short>(curX, curY);
				iSy = Sdy.at<short>(curX, curY);

				if ((iSx != 0 || iSy != 0) && (iTx != 0 || iTy != 0)) {
					partialSum += ((iSx * iTx) + (iSy * iTy)) * (edgeMagnitude[m] * matGradMag[curX][curY]);
				}
				//计算得分
				sumOfCoords = m + 1;
				partialScore = partialSum / sumOfCoords;

				if (partialScore < (MIN((minScore - 1) + normGreediness * sumOfCoords, normMinScore * sumOfCoords))) {
					break;
				}
			}
			result->at<float>(i, j) = partialScore;
		}
	}
	
	ReleaseDoubleMatrix(matGradMag, Ssize.height);
	Sdx.release();
	Sdy.release();
	return 1;
}


GeoMatch::~GeoMatch() {
	delete[] cordinates;
	delete[] edgeMagnitude;
	delete[] edgeDerivativeX;
	delete[] edgeDerivativeY;
}


void GeoMatch::CreateDoubleMatrix(double**& matrix, cv::Size size) {
	matrix = new double* [size.height];
	for (int iInd = 0; iInd < size.height; iInd++) {
		matrix[iInd] = new double[size.width];
	}
	return;
}


void GeoMatch::ReleaseDoubleMatrix(double**& matrix, int size) {
	for (int iInd = 0; iInd < size; iInd++) {
		delete[] matrix[iInd];
	}
	return;
}

cv::Mat GeoMatch::DrawMatch(cv::Mat pImage, cv::Size templSize, std::vector<GeoMatch::MatchResult> matchResultFilt, cv::Scalar color, int lineWidth) {
    cv::Mat result = pImage.clone();
	int len = matchResultFilt.size();
	double angle = 0.0;
	double value;
	cv::Point center;
	for (int i = 0; i < len; ++i) {
		center = matchResultFilt[i].bestLoc;
		angle = matchResultFilt[i].bestAngle;
        value = matchResultFilt[i].bestValue * 100;
		cv::RotatedRect box = cv::RotatedRect(center, templSize, -angle);
		cv::Point2f corner[4];
		box.points(corner);
        cv::putText(result, std::to_string(value), center, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2.2);
		for (int j = 0; j < 4; j++) {
            cv::line(result, corner[j], corner[(j + 1) % 4], color, lineWidth);
		}
	}
    return result;
}

cv::Mat GeoMatch::QImage2Mat(QImage &image)
{
    cv::Mat mat;
    switch(image.format())
    {
    case QImage::Format_Grayscale8: // 灰度图，每个像素点1个字节（8位）
        // Mat构造：行数，列数，存储结构，数据，step每行多少字节
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_ARGB32: // uint32存储0xAARRGGBB，pc一般小端存储低位在前，所以字节顺序就成了BGRA
    case QImage::Format_RGB32: // Alpha为FF
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888: // RR,GG,BB字节顺序存储
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        // opencv需要转为BGR的字节顺序
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    case QImage::Format_RGBA64: // uint64存储，顺序和Format_ARGB32相反，RGBA
        mat = cv::Mat(image.height(), image.width(), CV_16UC4, (void*)image.constBits(), image.bytesPerLine());
        // opencv需要转为BGRA的字节顺序
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        break;
    }
    return mat;
}

QImage GeoMatch::Mat2QImage(cv::Mat& mat){
    QImage image;
    switch(mat.type())
    {
    case CV_8UC1:
        // QImage构造：数据，宽度，高度，每行多少字节，存储结构
        image = QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        break;
    case CV_8UC3:
        image = QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        image = image.rgbSwapped(); // BRG转为RGB
        break;
    case CV_8UC4:
        image = QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        break;
    case CV_16UC4:
        image = QImage((const unsigned char*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGBA64);
        image = image.rgbSwapped(); // BRG转为RGB
        break;
    }
    return image;
}

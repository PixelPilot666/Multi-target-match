#include "Match.h"
#include "ui_Match.h"
#include "GeoMatch.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_selectTemplate_clicked()
{
    QFileDialog askForTemplate(this);
    if(askForTemplate.exec()==0) return;
    QStringList fileList = askForTemplate.selectedFiles();
    QString filename = fileList[0];
    if(!(templ.load(filename))){ //检测是否正确载入
        QMessageBox::information(this, tr("错误"), tr("打开图像失败!"));
        return;
    }
    ui->showTemplate->setPixmap(QPixmap::fromImage(templ));
    return;
}


void MainWindow::on_seletSrc_clicked()
{
    QFileDialog askForTemplate(this);
    if(askForTemplate.exec()==0) return;
    QStringList fileList = askForTemplate.selectedFiles();
    QString filename = fileList[0];
    if(!(src.load(filename))){//检测是否正确载入
        QMessageBox::information(this, tr("错误"), tr("打开图像失败!"));
        return;
    }
    ui->showSrc->setPixmap(QPixmap::fromImage(src));
    return;
}


QImage MainWindow::match(QImage templ, QImage src, double thresh, double minContrast, double maxContrast, double greediness, int pyramidLayers){
    GeoMatch geomatch;
    //预处理
    cv::Mat srcArr = geomatch.QImage2Mat(src);
    cv::Mat templArr = geomatch.QImage2Mat(templ);
    cv::Mat srcGray, templGray;
    //转灰度图
    cv::cvtColor(srcArr, srcGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(templArr, templGray, cv::COLOR_BGR2GRAY);
    cv::Point2f templCenter = cv::Point2f(templGray.cols / 2, templGray.rows / 2);
    //金字塔加速
    std::vector<GeoMatch::MatchResult> matchResultFilt = geomatch.PyramidMatching(srcGray, templGray, maxContrast, minContrast, thresh, greediness, pyramidLayers);
    resultMat = geomatch.DrawMatch(srcArr, templArr.size(), matchResultFilt, cv::Scalar(255, 255, 255), 2);
    QImage result = geomatch.Mat2QImage(resultMat);
    return result;
}



void MainWindow::on_startMatch_clicked()
{

    //判断是否为空
    if(src.isNull()){
        QMessageBox::information(this, tr("错误"), tr("未导入原图！"));
        return;
    }
    if(templ.isNull()){
        QMessageBox::information(this, tr("错误"), tr("未导入模板！"));
        return;
    }
    ui->state->setText(QString("Matching..."));
    QCoreApplication::processEvents(); // 强制刷新界面，以显示文本更新
    std::cout << "Match  start." << std::endl;

    //参数获取
    double thresh, minContrast, maxContrast, greediness;
    int pyramidLayers;
    thresh = ui->thresh->text().toDouble();
    minContrast = ui->minContrast->text().toDouble();
    maxContrast = ui->maxContrast->text().toDouble();
    greediness = ui->greediness->text().toDouble();
    pyramidLayers = ui->pyramidLayer->text().toInt();

    //匹配
    QImage result = match(templ, src, thresh, minContrast, maxContrast, greediness, pyramidLayers);

    ui->showSrc->setPixmap(QPixmap::fromImage(result));
    ui->state->setText(QString("Match End."));
    return;
}



void MainWindow::on_saveResult_clicked()
{
    if(resultMat.empty()){ //判断是否产生结果
        QMessageBox::information(this, tr("错误"), tr("未进行匹配！"));
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this, tr("Save Result"),"",tr("Image (*.jpg)"));
    std::string fileSave = filename.toStdString();
    cv::imwrite(fileSave, resultMat);
    ui->state->setText(QString("Save Done."));
    return;
}


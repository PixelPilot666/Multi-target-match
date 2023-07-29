#ifndef MATCH_H
#define MATCH_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QString>
#include <iostream>
#include <opencv2/core.hpp>


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_selectTemplate_clicked();

    void on_seletSrc_clicked();

    QImage match(QImage templ, QImage src, double thresh, double minContrast, double maxContrast, double greediness, int pyramidLayers);

    void on_startMatch_clicked();

    void on_saveResult_clicked();

private:
    Ui::MainWindow *ui;
    QImage templ, src;
    cv::Mat resultMat;
};
#endif // MATCH_H

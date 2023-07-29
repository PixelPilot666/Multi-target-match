#include "Match.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowFlags(w.windowFlags()&~Qt::WindowMinMaxButtonsHint|Qt::WindowMinimizeButtonHint); // 禁止最小化

    w.setWindowTitle(QString("模板匹配"));
    w.show();
    return a.exec();
}

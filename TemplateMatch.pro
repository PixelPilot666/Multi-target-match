QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    GeoMatch.cpp \
    Match.cpp \
    main.cpp

HEADERS += \
    GeoMatch.h \
    Match.h

FORMS += \
    Match.ui

INCLUDEPATH += D:\opencv3.4.16\opencv\opencv-build\install\include
LIBS += D:\opencv3.4.16\opencv\opencv-build\lib\libopencv_*.a


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

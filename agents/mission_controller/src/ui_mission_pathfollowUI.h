/********************************************************************************
** Form generated from reading UI file 'mission_pathfollowUI.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MISSION_PATHFOLLOWUI_H
#define UI_MISSION_PATHFOLLOWUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_PathFollow_UI
{
public:
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_4;
    QVBoxLayout *verticalLayout_5;
    QVBoxLayout *verticalLayout_4;
    QVBoxLayout *verticalLayout;
    QRadioButton *circle_radio_button;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QSlider *circle_radius_slider;
    QLCDNumber *circle_radius_lcdNumber;
    QVBoxLayout *verticalLayout_3;
    QRadioButton *oval_radio_button;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_2;
    QSlider *oval_short_radius_slider;
    QLCDNumber *oval_short_radius_lcdNumber;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_3;
    QSlider *oval_long_radius_slider;
    QLCDNumber *oval_long_radius_lcdNumber;
    QWidget *test_widget;
    QButtonGroup *button_group;

    void setupUi(QWidget *PathFollow_UI)
    {
        if (PathFollow_UI->objectName().isEmpty())
            PathFollow_UI->setObjectName(QString::fromUtf8("PathFollow_UI"));
        PathFollow_UI->resize(692, 331);
        verticalLayout_2 = new QVBoxLayout(PathFollow_UI);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        circle_radio_button = new QRadioButton(PathFollow_UI);
        button_group = new QButtonGroup(PathFollow_UI);
        button_group->setObjectName(QString::fromUtf8("button_group"));
        button_group->addButton(circle_radio_button);
        circle_radio_button->setObjectName(QString::fromUtf8("circle_radio_button"));
        circle_radio_button->setChecked(true);

        verticalLayout->addWidget(circle_radio_button);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(PathFollow_UI);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        circle_radius_slider = new QSlider(PathFollow_UI);
        circle_radius_slider->setObjectName(QString::fromUtf8("circle_radius_slider"));
        circle_radius_slider->setMaximum(1450);
        circle_radius_slider->setValue(0);
        circle_radius_slider->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(circle_radius_slider);

        circle_radius_lcdNumber = new QLCDNumber(PathFollow_UI);
        circle_radius_lcdNumber->setObjectName(QString::fromUtf8("circle_radius_lcdNumber"));
        circle_radius_lcdNumber->setProperty("intValue", QVariant(0));

        horizontalLayout->addWidget(circle_radius_lcdNumber);


        verticalLayout->addLayout(horizontalLayout);


        verticalLayout_4->addLayout(verticalLayout);


        verticalLayout_5->addLayout(verticalLayout_4);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        oval_radio_button = new QRadioButton(PathFollow_UI);
        button_group->addButton(oval_radio_button);
        oval_radio_button->setObjectName(QString::fromUtf8("oval_radio_button"));

        verticalLayout_3->addWidget(oval_radio_button);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_2 = new QLabel(PathFollow_UI);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_2->addWidget(label_2);

        oval_short_radius_slider = new QSlider(PathFollow_UI);
        oval_short_radius_slider->setObjectName(QString::fromUtf8("oval_short_radius_slider"));
        oval_short_radius_slider->setMaximum(1100);
        oval_short_radius_slider->setValue(0);
        oval_short_radius_slider->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(oval_short_radius_slider);

        oval_short_radius_lcdNumber = new QLCDNumber(PathFollow_UI);
        oval_short_radius_lcdNumber->setObjectName(QString::fromUtf8("oval_short_radius_lcdNumber"));
        oval_short_radius_lcdNumber->setProperty("intValue", QVariant(0));

        horizontalLayout_2->addWidget(oval_short_radius_lcdNumber);


        verticalLayout_3->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_3 = new QLabel(PathFollow_UI);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_3->addWidget(label_3);

        oval_long_radius_slider = new QSlider(PathFollow_UI);
        oval_long_radius_slider->setObjectName(QString::fromUtf8("oval_long_radius_slider"));
        oval_long_radius_slider->setMaximum(800);
        oval_long_radius_slider->setValue(0);
        oval_long_radius_slider->setOrientation(Qt::Horizontal);

        horizontalLayout_3->addWidget(oval_long_radius_slider);

        oval_long_radius_lcdNumber = new QLCDNumber(PathFollow_UI);
        oval_long_radius_lcdNumber->setObjectName(QString::fromUtf8("oval_long_radius_lcdNumber"));
        oval_long_radius_lcdNumber->setProperty("intValue", QVariant(0));

        horizontalLayout_3->addWidget(oval_long_radius_lcdNumber);


        verticalLayout_3->addLayout(horizontalLayout_3);


        verticalLayout_5->addLayout(verticalLayout_3);


        horizontalLayout_4->addLayout(verticalLayout_5);

        test_widget = new QWidget(PathFollow_UI);
        test_widget->setObjectName(QString::fromUtf8("test_widget"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(test_widget->sizePolicy().hasHeightForWidth());
        test_widget->setSizePolicy(sizePolicy);
        test_widget->setMinimumSize(QSize(0, 0));
        test_widget->setMaximumSize(QSize(400, 200));

        horizontalLayout_4->addWidget(test_widget);


        verticalLayout_2->addLayout(horizontalLayout_4);


        retranslateUi(PathFollow_UI);
        QObject::connect(circle_radius_slider, SIGNAL(valueChanged(int)), circle_radius_lcdNumber, SLOT(display(int)));
        QObject::connect(oval_short_radius_slider, SIGNAL(valueChanged(int)), oval_short_radius_lcdNumber, SLOT(display(int)));
        QObject::connect(oval_long_radius_slider, SIGNAL(valueChanged(int)), oval_long_radius_lcdNumber, SLOT(display(int)));

        QMetaObject::connectSlotsByName(PathFollow_UI);
    } // setupUi

    void retranslateUi(QWidget *PathFollow_UI)
    {
        PathFollow_UI->setWindowTitle(QApplication::translate("PathFollow_UI", "Follow Path Action", nullptr));
        circle_radio_button->setText(QApplication::translate("PathFollow_UI", "circle", nullptr));
        label->setText(QApplication::translate("PathFollow_UI", "radius", nullptr));
        oval_radio_button->setText(QApplication::translate("PathFollow_UI", "oval", nullptr));
        label_2->setText(QApplication::translate("PathFollow_UI", "short r", nullptr));
        label_3->setText(QApplication::translate("PathFollow_UI", "long r", nullptr));
    } // retranslateUi

};

namespace Ui {
    class PathFollow_UI: public Ui_PathFollow_UI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MISSION_PATHFOLLOWUI_H

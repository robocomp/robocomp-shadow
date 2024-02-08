/********************************************************************************
** Form generated from reading UI file 'localUI.ui'
**
** Created by: Qt User Interface Compiler version 6.2.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LOCALUI_H
#define UI_LOCALUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_local_guiDlg
{
public:
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_4;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLCDNumber *lcdNumber_hz;
    QPushButton *pushButton_stop;
    QSpacerItem *horizontalSpacer;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_2;
    QLCDNumber *lcdNumber_people;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_3;
    QLCDNumber *lcdNumber_room;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_4;
    QLCDNumber *lcdNumber_elapsed;
    QFrame *frame;

    void setupUi(QWidget *local_guiDlg)
    {
        if (local_guiDlg->objectName().isEmpty())
            local_guiDlg->setObjectName(QString::fromUtf8("local_guiDlg"));
        local_guiDlg->resize(800, 600);
        verticalLayout_2 = new QVBoxLayout(local_guiDlg);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(local_guiDlg);
        label->setObjectName(QString::fromUtf8("label"));
        label->setFrameShape(QFrame::StyledPanel);

        horizontalLayout->addWidget(label);

        lcdNumber_hz = new QLCDNumber(local_guiDlg);
        lcdNumber_hz->setObjectName(QString::fromUtf8("lcdNumber_hz"));
        QFont font;
        font.setFamilies({QString::fromUtf8("Ubuntu")});
        font.setPointSize(12);
        font.setBold(true);
        lcdNumber_hz->setFont(font);

        horizontalLayout->addWidget(lcdNumber_hz);


        horizontalLayout_4->addLayout(horizontalLayout);

        pushButton_stop = new QPushButton(local_guiDlg);
        pushButton_stop->setObjectName(QString::fromUtf8("pushButton_stop"));
        pushButton_stop->setCheckable(true);
        pushButton_stop->setChecked(true);

        horizontalLayout_4->addWidget(pushButton_stop);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_2 = new QLabel(local_guiDlg);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setFrameShape(QFrame::StyledPanel);

        horizontalLayout_2->addWidget(label_2);

        lcdNumber_people = new QLCDNumber(local_guiDlg);
        lcdNumber_people->setObjectName(QString::fromUtf8("lcdNumber_people"));
        QFont font1;
        font1.setBold(true);
        lcdNumber_people->setFont(font1);

        horizontalLayout_2->addWidget(lcdNumber_people);


        horizontalLayout_4->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_3 = new QLabel(local_guiDlg);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setFrameShape(QFrame::StyledPanel);

        horizontalLayout_3->addWidget(label_3);

        lcdNumber_room = new QLCDNumber(local_guiDlg);
        lcdNumber_room->setObjectName(QString::fromUtf8("lcdNumber_room"));
        lcdNumber_room->setFont(font1);

        horizontalLayout_3->addWidget(lcdNumber_room);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_4 = new QLabel(local_guiDlg);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFrameShape(QFrame::StyledPanel);

        horizontalLayout_5->addWidget(label_4);

        lcdNumber_elapsed = new QLCDNumber(local_guiDlg);
        lcdNumber_elapsed->setObjectName(QString::fromUtf8("lcdNumber_elapsed"));
        lcdNumber_elapsed->setFont(font1);

        horizontalLayout_5->addWidget(lcdNumber_elapsed);


        horizontalLayout_3->addLayout(horizontalLayout_5);


        horizontalLayout_4->addLayout(horizontalLayout_3);


        verticalLayout->addLayout(horizontalLayout_4);

        frame = new QFrame(local_guiDlg);
        frame->setObjectName(QString::fromUtf8("frame"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);

        verticalLayout->addWidget(frame);


        verticalLayout_2->addLayout(verticalLayout);


        retranslateUi(local_guiDlg);

        QMetaObject::connectSlotsByName(local_guiDlg);
    } // setupUi

    void retranslateUi(QWidget *local_guiDlg)
    {
        local_guiDlg->setWindowTitle(QCoreApplication::translate("local_guiDlg", "intention_predictor", nullptr));
        label->setText(QCoreApplication::translate("local_guiDlg", "Hz", nullptr));
        pushButton_stop->setText(QCoreApplication::translate("local_guiDlg", "STOP", nullptr));
        label_2->setText(QCoreApplication::translate("local_guiDlg", "People", nullptr));
        label_3->setText(QCoreApplication::translate("local_guiDlg", "Room", nullptr));
        label_4->setText(QCoreApplication::translate("local_guiDlg", "Elapsed", nullptr));
    } // retranslateUi

};

namespace Ui {
    class local_guiDlg: public Ui_local_guiDlg {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LOCALUI_H

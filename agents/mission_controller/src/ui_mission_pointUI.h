/********************************************************************************
** Form generated from reading UI file 'mission_pointUI.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MISSION_POINTUI_H
#define UI_MISSION_POINTUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Goto_UI
{
public:
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_2;
    QHBoxLayout *horizontalLayout;
    QLabel *labelX;
    QSpinBox *goto_spinbox_coordX;
    QLabel *labelY;
    QSpinBox *goto_spinbox_coordY;
    QLabel *label;
    QSpinBox *goto_spinbox_angle;
    QSpacerItem *horizontalSpacer;

    void setupUi(QWidget *Goto_UI)
    {
        if (Goto_UI->objectName().isEmpty())
            Goto_UI->setObjectName(QString::fromUtf8("Goto_UI"));
        Goto_UI->resize(846, 192);
        verticalLayout_2 = new QVBoxLayout(Goto_UI);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        labelX = new QLabel(Goto_UI);
        labelX->setObjectName(QString::fromUtf8("labelX"));

        horizontalLayout->addWidget(labelX);

        goto_spinbox_coordX = new QSpinBox(Goto_UI);
        goto_spinbox_coordX->setObjectName(QString::fromUtf8("goto_spinbox_coordX"));
        goto_spinbox_coordX->setMinimum(-13000);
        goto_spinbox_coordX->setMaximum(13000);

        horizontalLayout->addWidget(goto_spinbox_coordX);

        labelY = new QLabel(Goto_UI);
        labelY->setObjectName(QString::fromUtf8("labelY"));

        horizontalLayout->addWidget(labelY);

        goto_spinbox_coordY = new QSpinBox(Goto_UI);
        goto_spinbox_coordY->setObjectName(QString::fromUtf8("goto_spinbox_coordY"));
        goto_spinbox_coordY->setMinimum(-13000);
        goto_spinbox_coordY->setMaximum(13000);

        horizontalLayout->addWidget(goto_spinbox_coordY);

        label = new QLabel(Goto_UI);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        goto_spinbox_angle = new QSpinBox(Goto_UI);
        goto_spinbox_angle->setObjectName(QString::fromUtf8("goto_spinbox_angle"));

        horizontalLayout->addWidget(goto_spinbox_angle);


        horizontalLayout_2->addLayout(horizontalLayout);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout_2);


        verticalLayout_2->addLayout(verticalLayout);


        retranslateUi(Goto_UI);

        QMetaObject::connectSlotsByName(Goto_UI);
    } // setupUi

    void retranslateUi(QWidget *Goto_UI)
    {
        Goto_UI->setWindowTitle(QApplication::translate("Goto_UI", "Goto action", nullptr));
        labelX->setText(QApplication::translate("Goto_UI", "Coord X:", nullptr));
        labelY->setText(QApplication::translate("Goto_UI", "Coord Y:", nullptr));
        label->setText(QApplication::translate("Goto_UI", "Angle", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Goto_UI: public Ui_Goto_UI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MISSION_POINTUI_H

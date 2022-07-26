/********************************************************************************
** Form generated from reading UI file 'localUI.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LOCALUI_H
#define UI_LOCALUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_local_guiDlg
{
public:
    QVBoxLayout *verticalLayout_2;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout_5;
    QLabel *label;
    QComboBox *list_plan;
    QLabel *label_2;
    QListWidget *object_list;
    QPlainTextEdit *textedit_current_plan;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_4;
    QVBoxLayout *verticalLayout_3;
    QPushButton *pushButton_start_mission;
    QPushButton *pushButton_stop_mission;
    QPushButton *pushButton_cancel_mission;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QPushButton *path_trail_button;
    QSpacerItem *horizontalSpacer;
    QStackedWidget *stacked_widget;
    QWidget *goto_widget;
    QWidget *pathfollow_widget;

    void setupUi(QWidget *local_guiDlg)
    {
        if (local_guiDlg->objectName().isEmpty())
            local_guiDlg->setObjectName(QString::fromUtf8("local_guiDlg"));
        local_guiDlg->resize(994, 794);
        verticalLayout_2 = new QVBoxLayout(local_guiDlg);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        label = new QLabel(local_guiDlg);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);

        verticalLayout_5->addWidget(label);

        list_plan = new QComboBox(local_guiDlg);
        list_plan->setObjectName(QString::fromUtf8("list_plan"));

        verticalLayout_5->addWidget(list_plan);

        label_2 = new QLabel(local_guiDlg);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_5->addWidget(label_2);

        object_list = new QListWidget(local_guiDlg);
        object_list->setObjectName(QString::fromUtf8("object_list"));

        verticalLayout_5->addWidget(object_list);

        textedit_current_plan = new QPlainTextEdit(local_guiDlg);
        textedit_current_plan->setObjectName(QString::fromUtf8("textedit_current_plan"));
        textedit_current_plan->setEnabled(true);
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Maximum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(textedit_current_plan->sizePolicy().hasHeightForWidth());
        textedit_current_plan->setSizePolicy(sizePolicy1);

        verticalLayout_5->addWidget(textedit_current_plan);


        horizontalLayout_2->addLayout(verticalLayout_5);

        groupBox = new QGroupBox(local_guiDlg);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QFont font;
        font.setFamily(QString::fromUtf8("Ubuntu Mono"));
        font.setBold(false);
        font.setWeight(50);
        groupBox->setFont(font);
        groupBox->setLayoutDirection(Qt::LeftToRight);
        groupBox->setAlignment(Qt::AlignCenter);
        verticalLayout_4 = new QVBoxLayout(groupBox);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        pushButton_start_mission = new QPushButton(groupBox);
        pushButton_start_mission->setObjectName(QString::fromUtf8("pushButton_start_mission"));

        verticalLayout_3->addWidget(pushButton_start_mission);

        pushButton_stop_mission = new QPushButton(groupBox);
        pushButton_stop_mission->setObjectName(QString::fromUtf8("pushButton_stop_mission"));

        verticalLayout_3->addWidget(pushButton_stop_mission);

        pushButton_cancel_mission = new QPushButton(groupBox);
        pushButton_cancel_mission->setObjectName(QString::fromUtf8("pushButton_cancel_mission"));

        verticalLayout_3->addWidget(pushButton_cancel_mission);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);


        verticalLayout_4->addLayout(verticalLayout_3);


        horizontalLayout_2->addWidget(groupBox);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        path_trail_button = new QPushButton(local_guiDlg);
        path_trail_button->setObjectName(QString::fromUtf8("path_trail_button"));
        path_trail_button->setCheckable(true);
        path_trail_button->setChecked(true);

        horizontalLayout->addWidget(path_trail_button);

        horizontalSpacer = new QSpacerItem(108, 17, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout);

        stacked_widget = new QStackedWidget(local_guiDlg);
        stacked_widget->setObjectName(QString::fromUtf8("stacked_widget"));
        stacked_widget->setFrameShape(QFrame::Box);
        goto_widget = new QWidget();
        goto_widget->setObjectName(QString::fromUtf8("goto_widget"));
        stacked_widget->addWidget(goto_widget);
        pathfollow_widget = new QWidget();
        pathfollow_widget->setObjectName(QString::fromUtf8("pathfollow_widget"));
        stacked_widget->addWidget(pathfollow_widget);

        verticalLayout->addWidget(stacked_widget);


        verticalLayout_2->addLayout(verticalLayout);


        retranslateUi(local_guiDlg);

        QMetaObject::connectSlotsByName(local_guiDlg);
    } // setupUi

    void retranslateUi(QWidget *local_guiDlg)
    {
        local_guiDlg->setWindowTitle(QApplication::translate("local_guiDlg", "Giraff Mission Controller", nullptr));
        label->setText(QApplication::translate("local_guiDlg", "Current mission", nullptr));
        label_2->setText(QApplication::translate("local_guiDlg", "Element to follow", nullptr));
        groupBox->setTitle(QApplication::translate("local_guiDlg", "Control", nullptr));
        pushButton_start_mission->setText(QApplication::translate("local_guiDlg", "Start", nullptr));
        pushButton_stop_mission->setText(QApplication::translate("local_guiDlg", "Stop", nullptr));
        pushButton_cancel_mission->setText(QApplication::translate("local_guiDlg", "Cancel", nullptr));
        path_trail_button->setText(QApplication::translate("local_guiDlg", "path trail", nullptr));
    } // retranslateUi

};

namespace Ui {
    class local_guiDlg: public Ui_local_guiDlg {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LOCALUI_H

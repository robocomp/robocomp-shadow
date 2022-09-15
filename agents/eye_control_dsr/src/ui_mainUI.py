# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_guiDlg(object):
    def setupUi(self, guiDlg):
        if not guiDlg.objectName():
            guiDlg.setObjectName(u"guiDlg")
        guiDlg.resize(662, 992)
        self.verticalLayout_7 = QVBoxLayout(guiDlg)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSlider_pos = QSlider(guiDlg)
        self.horizontalSlider_pos.setObjectName(u"horizontalSlider_pos")
        self.horizontalSlider_pos.setLayoutDirection(Qt.LeftToRight)
        self.horizontalSlider_pos.setMinimum(-90)
        self.horizontalSlider_pos.setMaximum(90)
        self.horizontalSlider_pos.setOrientation(Qt.Horizontal)

        self.horizontalLayout_3.addWidget(self.horizontalSlider_pos)

        self.lcdNumber_pos = QLCDNumber(guiDlg)
        self.lcdNumber_pos.setObjectName(u"lcdNumber_pos")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lcdNumber_pos.setFont(font)

        self.horizontalLayout_3.addWidget(self.lcdNumber_pos)

        self.label_position = QLabel(guiDlg)
        self.label_position.setObjectName(u"label_position")
        font1 = QFont()
        font1.setPointSize(11)
        self.label_position.setFont(font1)

        self.horizontalLayout_3.addWidget(self.label_position)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSlider_max_speed = QSlider(guiDlg)
        self.horizontalSlider_max_speed.setObjectName(u"horizontalSlider_max_speed")
        self.horizontalSlider_max_speed.setLayoutDirection(Qt.LeftToRight)
        self.horizontalSlider_max_speed.setMaximum(1023)
        self.horizontalSlider_max_speed.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.horizontalSlider_max_speed)

        self.lcdNumber_max_speed = QLCDNumber(guiDlg)
        self.lcdNumber_max_speed.setObjectName(u"lcdNumber_max_speed")
        self.lcdNumber_max_speed.setFont(font)

        self.horizontalLayout_2.addWidget(self.lcdNumber_max_speed)

        self.label_max_speed = QLabel(guiDlg)
        self.label_max_speed.setObjectName(u"label_max_speed")
        self.label_max_speed.setFont(font1)

        self.horizontalLayout_2.addWidget(self.label_max_speed)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalSlider = QSlider(guiDlg)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setMinimum(-600)
        self.horizontalSlider.setMaximum(600)
        self.horizontalSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_8.addWidget(self.horizontalSlider)

        self.lcdNumber = QLCDNumber(guiDlg)
        self.lcdNumber.setObjectName(u"lcdNumber")

        self.horizontalLayout_8.addWidget(self.lcdNumber)

        self.label = QLabel(guiDlg)
        self.label.setObjectName(u"label")

        self.horizontalLayout_8.addWidget(self.label)


        self.verticalLayout_2.addLayout(self.horizontalLayout_8)


        self.verticalLayout_5.addLayout(self.verticalLayout_2)


        self.verticalLayout_6.addLayout(self.verticalLayout_5)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.lcdNumber_speed = QLCDNumber(guiDlg)
        self.lcdNumber_speed.setObjectName(u"lcdNumber_speed")

        self.verticalLayout_3.addWidget(self.lcdNumber_speed)

        self.label_speed = QLabel(guiDlg)
        self.label_speed.setObjectName(u"label_speed")
        self.label_speed.setFont(font1)
        self.label_speed.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_speed)


        self.horizontalLayout_4.addLayout(self.verticalLayout_3)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.lcdNumber_temp = QLCDNumber(guiDlg)
        self.lcdNumber_temp.setObjectName(u"lcdNumber_temp")

        self.verticalLayout_4.addWidget(self.lcdNumber_temp)

        self.label_temperature = QLabel(guiDlg)
        self.label_temperature.setObjectName(u"label_temperature")
        self.label_temperature.setFont(font1)
        self.label_temperature.setLayoutDirection(Qt.LeftToRight)
        self.label_temperature.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_temperature)


        self.horizontalLayout_4.addLayout(self.verticalLayout_4)


        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pushButton_center = QPushButton(guiDlg)
        self.pushButton_center.setObjectName(u"pushButton_center")

        self.horizontalLayout.addWidget(self.pushButton_center)

        self.radioButton_moving = QRadioButton(guiDlg)
        self.radioButton_moving.setObjectName(u"radioButton_moving")

        self.horizontalLayout.addWidget(self.radioButton_moving)


        self.horizontalLayout_5.addLayout(self.horizontalLayout)


        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.pushButton = QPushButton(guiDlg)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setCheckable(True)

        self.verticalLayout.addWidget(self.pushButton)

        self.pushButton_2 = QPushButton(guiDlg)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setStyleSheet(u"background-color: rgb(204, 0, 0);")

        self.verticalLayout.addWidget(self.pushButton_2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_6.addLayout(self.verticalLayout)

        self.frame_error = QFrame(guiDlg)
        self.frame_error.setObjectName(u"frame_error")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_error.sizePolicy().hasHeightForWidth())
        self.frame_error.setSizePolicy(sizePolicy)
        self.frame_error.setMinimumSize(QSize(0, 200))
        self.frame_error.setFrameShape(QFrame.Panel)

        self.horizontalLayout_6.addWidget(self.frame_error)


        self.verticalLayout_6.addLayout(self.horizontalLayout_6)

        self.label_image = QLabel(guiDlg)
        self.label_image.setObjectName(u"label_image")
        self.label_image.setMinimumSize(QSize(480, 640))

        self.verticalLayout_6.addWidget(self.label_image)


        self.verticalLayout_7.addLayout(self.verticalLayout_6)


        self.retranslateUi(guiDlg)

        QMetaObject.connectSlotsByName(guiDlg)
    # setupUi

    def retranslateUi(self, guiDlg):
        guiDlg.setWindowTitle(QCoreApplication.translate("guiDlg", u"Eye Control", None))
        self.label_position.setText(QCoreApplication.translate("guiDlg", u"position (rad)", None))
        self.label_max_speed.setText(QCoreApplication.translate("guiDlg", u"max vel (rad/s)", None))
        self.label.setText(QCoreApplication.translate("guiDlg", u"Ver giraff speed(rad/s)", None))
        self.label_speed.setText(QCoreApplication.translate("guiDlg", u" speed (rads/s)", None))
        self.label_temperature.setText(QCoreApplication.translate("guiDlg", u"temp \u00baC", None))
        self.pushButton_center.setText(QCoreApplication.translate("guiDlg", u"center", None))
        self.radioButton_moving.setText(QCoreApplication.translate("guiDlg", u"Moving", None))
        self.pushButton.setText(QCoreApplication.translate("guiDlg", u"track", None))
        self.pushButton_2.setText(QCoreApplication.translate("guiDlg", u"STOP", None))
        self.label_image.setText(QCoreApplication.translate("guiDlg", u"TextLabel", None))
    # retranslateUi


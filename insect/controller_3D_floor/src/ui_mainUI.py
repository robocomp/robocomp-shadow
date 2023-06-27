# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_guiDlg(object):
    def setupUi(self, guiDlg):
        if not guiDlg.objectName():
            guiDlg.setObjectName(u"guiDlg")
        guiDlg.resize(848, 865)
        guiDlg.setMaximumSize(QSize(16777215, 16777215))
        self.verticalLayout_2 = QVBoxLayout(guiDlg)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.pushButton_leftleft = QPushButton(guiDlg)
        self.pushButton_leftleft.setObjectName(u"pushButton_leftleft")

        self.horizontalLayout.addWidget(self.pushButton_leftleft)

        self.pushButton_left = QPushButton(guiDlg)
        self.pushButton_left.setObjectName(u"pushButton_left")
        self.pushButton_left.setCheckable(False)

        self.horizontalLayout.addWidget(self.pushButton_left)

        self.pushButton_centre = QPushButton(guiDlg)
        self.pushButton_centre.setObjectName(u"pushButton_centre")
        self.pushButton_centre.setCheckable(False)

        self.horizontalLayout.addWidget(self.pushButton_centre)

        self.pushButton_right = QPushButton(guiDlg)
        self.pushButton_right.setObjectName(u"pushButton_right")
        self.pushButton_right.setCheckable(False)

        self.horizontalLayout.addWidget(self.pushButton_right)

        self.pushButton_rightright = QPushButton(guiDlg)
        self.pushButton_rightright.setObjectName(u"pushButton_rightright")

        self.horizontalLayout.addWidget(self.pushButton_rightright)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.frame_2d = QLabel(guiDlg)
        self.frame_2d.setObjectName(u"frame_2d")
        self.frame_2d.setMinimumSize(QSize(600, 300))

        self.verticalLayout.addWidget(self.frame_2d)

        self.frame_3d = QWidget(guiDlg)
        self.frame_3d.setObjectName(u"frame_3d")

        self.verticalLayout.addWidget(self.frame_3d)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(guiDlg)

        QMetaObject.connectSlotsByName(guiDlg)
    # setupUi

    def retranslateUi(self, guiDlg):
        guiDlg.setWindowTitle(QCoreApplication.translate("guiDlg", u"Insect Controller", None))
        self.pushButton_leftleft.setText(QCoreApplication.translate("guiDlg", u"leftleft", None))
        self.pushButton_left.setText(QCoreApplication.translate("guiDlg", u"left", None))
        self.pushButton_centre.setText(QCoreApplication.translate("guiDlg", u"centre", None))
        self.pushButton_right.setText(QCoreApplication.translate("guiDlg", u"right", None))
        self.pushButton_rightright.setText(QCoreApplication.translate("guiDlg", u"rightright", None))
        self.frame_2d.setText(QCoreApplication.translate("guiDlg", u"TextLabel", None))
    # retranslateUi


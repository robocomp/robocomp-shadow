# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUI.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QLabel, QListView,
    QPushButton, QSizePolicy, QWidget)

class Ui_guiDlg(object):
    def setupUi(self, guiDlg):
        if not guiDlg.objectName():
            guiDlg.setObjectName(u"guiDlg")
        guiDlg.resize(281, 628)
        self.people_list = QListView(guiDlg)
        self.people_list.setObjectName(u"people_list")
        self.people_list.setGeometry(QRect(20, 190, 241, 121))
        self.mission_list = QListView(guiDlg)
        self.mission_list.setObjectName(u"mission_list")
        self.mission_list.setGeometry(QRect(20, 40, 241, 111))
        self.label = QLabel(guiDlg)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 10, 261, 611))
        self.label.setStyleSheet(u"background-color: rgb(87, 227, 137);")
        self.label_2 = QLabel(guiDlg)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(100, 20, 67, 17))
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_3 = QLabel(guiDlg)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(110, 159, 67, 17))
        self.label_3.setAlignment(Qt.AlignCenter)
        self.execute_button = QPushButton(guiDlg)
        self.execute_button.setObjectName(u"execute_button")
        self.execute_button.setGeometry(QRect(20, 540, 71, 31))
        self.abort_button = QPushButton(guiDlg)
        self.abort_button.setObjectName(u"abort_button")
        self.abort_button.setGeometry(QRect(100, 580, 89, 25))
        self.abort_button.setStyleSheet(u"background-color: rgb(224, 27, 36);")
        self.label_4 = QLabel(guiDlg)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(60, 320, 161, 20))
        self.label_4.setAlignment(Qt.AlignCenter)
        self.destinations_list = QListView(guiDlg)
        self.destinations_list.setObjectName(u"destinations_list")
        self.destinations_list.setGeometry(QRect(20, 351, 241, 121))
        self.wait_button = QPushButton(guiDlg)
        self.wait_button.setObjectName(u"wait_button")
        self.wait_button.setGeometry(QRect(100, 540, 81, 31))
        self.stop_button = QPushButton(guiDlg)
        self.stop_button.setObjectName(u"stop_button")
        self.stop_button.setGeometry(QRect(190, 540, 71, 31))
        self.submission_check = QCheckBox(guiDlg)
        self.submission_check.setObjectName(u"submission_check")
        self.submission_check.setGeometry(QRect(20, 500, 101, 23))
        self.priority_check = QCheckBox(guiDlg)
        self.priority_check.setObjectName(u"priority_check")
        self.priority_check.setGeometry(QRect(150, 500, 101, 23))
        self.label.raise_()
        self.people_list.raise_()
        self.mission_list.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.execute_button.raise_()
        self.abort_button.raise_()
        self.label_4.raise_()
        self.destinations_list.raise_()
        self.wait_button.raise_()
        self.stop_button.raise_()
        self.submission_check.raise_()
        self.priority_check.raise_()

        self.retranslateUi(guiDlg)

        QMetaObject.connectSlotsByName(guiDlg)
    # setupUi

    def retranslateUi(self, guiDlg):
        guiDlg.setWindowTitle(QCoreApplication.translate("guiDlg", u"mission_controller_python", None))
        self.label.setText("")
        self.label_2.setText(QCoreApplication.translate("guiDlg", u"Missions", None))
        self.label_3.setText(QCoreApplication.translate("guiDlg", u"People", None))
        self.execute_button.setText(QCoreApplication.translate("guiDlg", u"Execute", None))
        self.abort_button.setText(QCoreApplication.translate("guiDlg", u"ABORT", None))
        self.label_4.setText(QCoreApplication.translate("guiDlg", u"Destinations", None))
        self.wait_button.setText(QCoreApplication.translate("guiDlg", u"Wait", None))
        self.stop_button.setText(QCoreApplication.translate("guiDlg", u"Stop", None))
        self.submission_check.setText(QCoreApplication.translate("guiDlg", u"Submission", None))
        self.priority_check.setText(QCoreApplication.translate("guiDlg", u"Priority", None))
    # retranslateUi


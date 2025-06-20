/*
 *    Copyright (C) 2020 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef CUSTOMWIDGET_H
#define CUSTOMWIDGET_H

#if Qt5_FOUND
	#include <QtWidgets>
#else
	#include <QtGui>
#endif

#include <QWidget>
#include "ui_localUI.h"
//#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

class CustomWidget : public QWidget
{
    Q_OBJECT

    public:
        CustomWidget(QWidget *parent = nullptr) : QWidget(parent), ui(new Ui::local_guiDlg)
        {
            ui->setupUi(this);
//            viewer = new AbstractGraphicViewer(ui->frame, QRectF(-6000, -6000, 12000, 12000));
//            viewer->add_robot(460, 480, 0, 100, QColor("Blue"));
//            viewer->show();
        }
        ~CustomWidget(){ delete ui; };
        Ui::local_guiDlg *ui;
//        AbstractGraphicViewer *viewer;
};

//class Custom_widget : public AbstractGraphicViewer
//{
//    Q_OBJECT
//    public:
//        Custom_widget() : AbstractGraphicViewer(nullptr, QRectF(-6000, -6000, 12000, 12000))
//        {
//              add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
//        }
//        ~Custom_widget()
//        { }
//
//        struct Params
//        {
//            float ROBOT_WIDTH = 460;  // mm
//            float ROBOT_LENGTH = 480;  // mm
//            QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
//        };
//        Params params;
//};
#endif

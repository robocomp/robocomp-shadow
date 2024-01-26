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

#include <ui_localUI.h>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>


class Custom_widget : public QWidget, public Ui_local_guiDlg
{
Q_OBJECT
public:
    Custom_widget() : Ui_local_guiDlg()
    {
        setupUi(this);
        // Viewer
//        viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
//        //QRectF(params.xMin, params.yMin, params.grid_width, params.grid_length));
//        viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
//        viewer->show();
    }
	~Custom_widget()
    {

    }
    //Graphics
    AbstractGraphicViewer *viewer;

    struct Params
    {
        float ROBOT_WIDTH = 460;  // mm
        float ROBOT_LENGTH = 480;  // mm
        QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
    };
    Params params;



};
#endif

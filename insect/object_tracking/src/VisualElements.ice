//******************************************************************
// 
//  Generated by RoboCompDSL
//  
//  File name: VisualElements.ice
//  Source: VisualElements.idsl
//
//******************************************************************
#ifndef ROBOCOMPVISUALELEMENTS_ICE
#define ROBOCOMPVISUALELEMENTS_ICE
#include <Person.ice>
#include <Camera360RGB.ice>
module RoboCompVisualElements
{
	struct TRoi
	{
		int xcenter;
		int ycenter;
		int xsize;
		int ysize;
		int finalxsize;
		int finalysize;
	};
	struct TObject
	{
		int id;
		int type;
		int left;
		int top;
		int right;
		int bot;
		float score;
		float depth;
		float x;
		float y;
		float z;
		RoboCompCamera360RGB::TImage image;
		RoboCompPerson::TPerson person;
	};
	sequence <TObject> TObjects;
	interface VisualElements
	{
		TObjects getVisualObjects (TObjects objects);
	};
};

#endif
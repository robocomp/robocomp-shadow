//******************************************************************
// 
//  Generated by RoboCompDSL
//  
//  File name: SegmentatorTrackingPub.ice
//  Source: SegmentatorTrackingPub.idsl
//
//******************************************************************
#ifndef ROBOCOMPSEGMENTATORTRACKINGPUB_ICE
#define ROBOCOMPSEGMENTATORTRACKINGPUB_ICE
#include <VisualElements.ice>
module RoboCompSegmentatorTrackingPub
{
	interface SegmentatorTrackingPub
	{
		idempotent void setTrack (RoboCompVisualElements::TObject target);
	};
};

#endif
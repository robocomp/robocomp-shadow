//
// Created by pbustos on 1/12/22.
//

#ifndef FORCEFIELD_EYE_TRACKER_H
#define FORCEFIELD_EYE_TRACKER_H

#include "robot.h"

namespace rc
{
    class Eye_Tracker
    {
        struct Params
        {
            float max_hor_angle_error = 0.6; // rads
        };
        Params consts;
        public:
            float track(rc::Robot &robot);
    };

}
#endif //FORCEFIELD_EYE_TRACKER_H

#include <cmath>
#include <Eigen/Core>
#include "Gaussian.hpp"
#include "StateSLAM.h"
#include "StateSLAMPoseLandmarks.h"
#include <iostream>

StateSLAMPoseLandmarks::StateSLAMPoseLandmarks(const Gaussian<double> & density)
    : StateSLAM(density)
{
    /*
    SQ_(0,0) = 10.0;
    SQ_(1,1) = 10.0;
    SQ_(2,2) = 5.0;
    SQ_(3,3) = 1.0;
    SQ_(4,4) = 1.0;
    SQ_(5,5) = 1.0;
    */
    /*
    SQ_(0,0) = 10.0;
    SQ_(1,1) = 10.0;
    SQ_(2,2) = 5.0;
    SQ_(3,3) = 0.5;
    SQ_(4,4) = 0.5;
    SQ_(5,5) = 1.0;
    */
    /*
    SQ_(0,0) = 0.5;
    SQ_(1,1) = 0.5;
    SQ_(2,2) = 0.1;
    SQ_(3,3) = 0.5;
    SQ_(4,4) = 0.5;
    SQ_(5,5) = 1.0;
    */
    SQ_(0,0) = 0.5;
    SQ_(1,1) = 0.5;
    SQ_(2,2) = 0.1;
    SQ_(3,3) = 0.01;
    SQ_(4,4) = 0.01;
    SQ_(5,5) = 0.2;
}

StateSLAM * StateSLAMPoseLandmarks::clone() const
{
    return new StateSLAMPoseLandmarks(*this);
}

std::size_t StateSLAMPoseLandmarks::numberLandmarks() const
{
    return (size() - 12)/6;
}

std::size_t StateSLAMPoseLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 6*idxLandmark;    
}
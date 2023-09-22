#include <cmath>
#include <Eigen/Core>
#include "Gaussian.hpp"
#include "StateSLAM.h"
#include "StateSLAMPoseLandmarks.h"
#include <iostream>

StateSLAMPoseLandmarks::StateSLAMPoseLandmarks(const Gaussian<double> & density)
    : StateSLAM(density)
{
    //SQ_.resize(density.sqrtCov().rows(),density.sqrtCov().cols());
    //SQ_.fill(0);
    /*
    Eigen::Vector6d tuningSQ_(10,10,10,1,1,1);
    for(int i = 0; i < tuningSQ_.size(); i++){
        SQ_(i,i) = tuningSQ_(i);
    }
    */
    SQ_(0,0) = 10.0;
    SQ_(1,1) = 10.0;
    SQ_(2,2) = 10.0;
    SQ_(3,3) = 1.0;
    SQ_(4,4) = 1.0;
    SQ_(5,5) = 1.0;
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
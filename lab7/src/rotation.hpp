#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>
#include <iostream>

template<typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    
    R(1, 1) = cos(x);
    R(2, 1) = sin(x);

    R(1, 2) = -sin(x);
    R(2, 2) = cos(x);

    return R;
}

template<typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx            =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(1,1)       = -sin(x);
    dRdx(2,1)       =  cos(x);

    dRdx(1,2)       = -cos(x);
    dRdx(2,2)       = -sin(x);
    return rotx(x);
}

template<typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();

    R(0,0)    =  cos(x);
    R(2,0)    = -sin(x);

    R(0,2)    =  sin(x);
    R(2,2)    =  cos(x);

    return R;
}

template<typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(2,0)    = -cos(x);

    dRdx(0,2)    =  cos(x);
    dRdx(2,2)    = -sin(x);
    return roty(x);
}

template<typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    
    R(0,0)    =  cos(x);
    R(1,0)    =  sin(x);

    R(0,1)    = -sin(x);
    R(1,1)    =  cos(x);

    return R;
}

template<typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(1,0)    =  cos(x);

    dRdx(0,1)    = -cos(x);
    dRdx(1,1)    = -sin(x);
    return rotz(x);
}

template<typename Scalar>
Eigen::Matrix3<Scalar> rpy2rot(const Eigen::Vector3<Scalar> & Theta)
{
    // R = Rz*Ry*Rx
    Eigen::Matrix3<Scalar> R;
    
    R = rotz(Theta(2))*roty(Theta(1))*rotx(Theta(0));
    return R;
}

template<typename Scalar>
Eigen::Vector3<Scalar> rot2rpy(const Eigen::Matrix3<Scalar> & R)
{
    using std::atan2, std::hypot;
    Eigen::Vector3<Scalar> Theta;

    Theta(2) = atan2(R(1,0),R(0,0));
    Theta(1) = atan2(-R(2,0),hypot(R(2,1),R(2,2)));
    Theta(0) = atan2(R(2,1),R(2,2));

    return Theta;
}

#endif

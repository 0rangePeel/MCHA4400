#ifndef MEASUREMENTPOSEBUNDLE_H
#define MEASUREMENTPOSEBUNDLE_H

#include <Eigen/Core>
#include "State.h"
#include "Camera.h"
#include "Measurement.h"
#include "StateSLAM.h"
#include "StateSLAMPoseLandmarks.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

class MeasurementPoseBundle : public Measurement
{
public:
    MeasurementPoseBundle(double time, const Eigen::VectorXd & y, const Camera & camera);
    virtual double logLikelihood(const State & state, const Eigen::VectorXd & x) const override;
    virtual double logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    //virtual double logLikelihood(const State & state, const Eigen::VectorXd & x, std::size_t idxLandmark, const int j, Eigen::VectorXd & g) const;
    //virtual double logLikelihood(const State & state, const Eigen::VectorXd & x, std::size_t idxLandmark, const int j, Eigen::VectorXd & g, Eigen::MatrixXd & H) const;

protected:
    virtual void update(State & state) override;
    const Camera & camera_;
    //template <typename Scalar> Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x, const Camera & cam, const StateSLAM stateSLAM, std::size_t idxLandmark, const int j) const;
};
/*
template <typename Scalar>
Scalar MeasurementPoseBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x, const Camera & cam, const StateSLAM stateSLAM, std::size_t idxLandmark, const int j) const
{
    Eigen::VectorX<Scalar> y = y_.cast<Scalar>();
    Eigen::VectorX<Scalar> h = stateSLAM.predictFeatureTag(x, cam, idxLandmark, j);
    Eigen::MatrixX<Scalar> SR = noise_.sqrtCov().cast<Scalar>();
    Gaussian<Scalar> likelihood(h, SR);
    return likelihood.log(y);
}
*/
#endif
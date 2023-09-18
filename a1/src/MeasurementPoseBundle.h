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

protected:
    virtual void update(State & state) override;
    const Camera & camera_;
    template <typename Scalar> Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x, const StateSLAM & stateSLAM, const std::vector <std::size_t> & idxLandmark) const;
};

template <typename Scalar>
Scalar MeasurementPoseBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x, const StateSLAM & stateSLAM, const std::vector <std::size_t> & idxLandmarks) const
{
    Eigen::VectorX<Scalar> y = y_.cast<Scalar>();
    Eigen::VectorX<Scalar> h = stateSLAM.predictFeatureTagBundle<Scalar>(x, camera_, idxLandmarks);
    Eigen::MatrixX<Scalar> SR = noise_.sqrtCov().cast<Scalar>();
    Gaussian<Scalar> likelihood(h, SR);
    return likelihood.log(y);
}

#endif
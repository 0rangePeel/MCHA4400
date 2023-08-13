#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cstddef>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>

class Gaussian
{
public:
    Gaussian();
    explicit Gaussian(std::size_t n);
    explicit Gaussian(const Eigen::MatrixXd & S);
    Gaussian(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S);

    std::size_t size() const;
    const Eigen::VectorXd & mean() const;
    const Eigen::MatrixXd & sqrtCov() const;
    Eigen::MatrixXd cov() const;

    // Joint distribution from product of independent Gaussians
    Gaussian operator*(const Gaussian & other) const;

    // Simulate (generate samples)
    Eigen::VectorXd simulate() const;

    // Marginal distribution
    template <typename IndexType> Gaussian marginal(const IndexType & idx) const;

    // Conditional distribution
    template <typename IndexTypeA, typename IndexTypeB> Gaussian conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd &xB) const;

    // Affine transform
    template <typename Func> Gaussian transform(Func h) const;
    template <typename Func> Gaussian transformWithAdditiveNoise(Func h, const Gaussian & noise) const;
protected:
    Eigen::VectorXd mu_;
    Eigen::MatrixXd S_;
};

// Given joint density p(x), return marginal density p(x(idx))
template <typename IndexType>
Gaussian Gaussian::marginal(const IndexType & idx) const
{
    Gaussian out(idx.size());
    // TODO
    // out.mu_ = ???;
    // out.S_ = ???;
    return out;
}

// Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
template <typename IndexTypeA, typename IndexTypeB>
Gaussian Gaussian::conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd & xB) const
{
    // FIXME: The following implementation is in error, but it does pass some of the unit tests
    Gaussian out;
    out.mu_ = mu_(idxA) +
        S_(idxB, idxA).transpose()*
        S_(idxB, idxB).eval().template triangularView<Eigen::Upper>().transpose().solve(xB - mu_(idxB));
    out.S_ = S_(idxA, idxA);
    return out;
}

template <typename Func>
Gaussian Gaussian::transform(Func h) const
{
    Gaussian out;
    Eigen::MatrixXd C;
    out.mu_ = h(mu_, C);
    const std::size_t & ny = out.mu_.rows();
    Eigen::MatrixXd SS = S_*C.transpose();
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(SS);   // In-place QR decomposition
    out.S_ = SS.topRows(ny).triangularView<Eigen::Upper>();
    return out;
}

template <typename Func>
Gaussian Gaussian::transformWithAdditiveNoise(Func h, const Gaussian & noise) const
{
    assert(noise.mean().isZero());
    Gaussian out;
    Eigen::MatrixXd C;
    out.mu_ = h(mu_, C) /*+ noise.mean()*/;
    const std::size_t & nx = mu_.rows();
    const std::size_t & ny = out.mu_.rows();
    Eigen::MatrixXd SS(nx + ny, ny);
    SS << S_*C.transpose(), noise.sqrtCov();
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(SS);   // In-place QR decomposition
    out.S_ = SS.topRows(ny).triangularView<Eigen::Upper>();
    return out;
}

#endif

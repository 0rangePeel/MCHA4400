#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <cstddef>
#include <cmath>
#include <ctime>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/LU> // for .determinant() and .inverse(), which you should never use

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

template <typename Scalar = double>
class Gaussian
{
public:
    Gaussian()
    {}

    explicit Gaussian(std::size_t n)
        : mu_(n)
        , S_(n, n)
    {}

    // template <typename OtherScalar>
    explicit Gaussian(const Eigen::MatrixX<Scalar> & S)
        : mu_(Eigen::VectorX<Scalar>::Zero(S.cols()))
        , S_(S)
    {
        assert(S_.rows() == S_.cols());
    }

    template <typename OtherScalar>
    Gaussian(const Eigen::VectorX<OtherScalar> & mu, const Eigen::MatrixX<OtherScalar> & S)
        : mu_(mu.template cast<Scalar>())
        , S_(S.template cast<Scalar>())
    {
        assert(S_.rows() == S_.cols());
        assert(mu_.rows() == S_.cols());
    }

    template <typename OtherScalar> friend class Gaussian;

    template <typename OtherScalar>
    Gaussian(const Gaussian<OtherScalar> & p)
        : mu_(p.mu_.template cast<Scalar>())
        , S_(p.S_.template cast<Scalar>())
    {
        assert(S_.rows() == S_.cols());
        assert(mu_.rows() == S_.cols());
    }

    template <typename OtherScalar>
    Gaussian<OtherScalar> cast() const
    {
        return Gaussian<OtherScalar>(*this);
    }

    Eigen::Index size() const
    {
        return mu_.size();
    }

    Eigen::VectorX<Scalar> & mean()
    {
        return mu_;
    }

    Eigen::MatrixX<Scalar> & sqrtCov()
    {
        return S_;
    }

    const Eigen::VectorX<Scalar> & mean() const
    {
        return mu_;
    }

    const Eigen::MatrixX<Scalar> & sqrtCov() const
    {
        return S_;
    }

    Eigen::MatrixX<Scalar> cov() const
    {
        return S_.transpose()*S_;
    }

    // Given joint density p(x), return marginal density p(x(idx))
    template <typename IndexType>
    Gaussian marginal(const IndexType & idx) const
    {
        Gaussian out;
        // out.mu_ = ???
        // out.S_ = ???
        return out;
    }

    // Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
    template <typename IndexTypeA, typename IndexTypeB>
    Gaussian conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorX<Scalar> & xB) const
    {
        // FIXME: The following implementation is in error, but it does pass some of the unit tests
        Gaussian out;
        out.mu_ = mu_(idxA) +
            S_(idxB, idxA).transpose()*
            S_(idxB, idxB).eval().template triangularView<Eigen::Upper>().transpose().solve(xB - mu_(idxB));
        out.S_ = S_(idxA,idxA);
        return out;
    }

    template <typename Func>
    Gaussian transform(Func h) const
    {
        Gaussian out;
        Eigen::MatrixX<Scalar> C;
        out.mu_ = h(mu_, C);
        const std::size_t & ny = out.mu_.rows();
        Eigen::MatrixX<Scalar> SS = S_*C.transpose();
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SS);   // In-place QR decomposition
        out.S_ = SS.topRows(ny).template triangularView<Eigen::Upper>();
        return out;
    }

    template <typename Func>
    Gaussian transformWithAdditiveNoise(Func h, const Gaussian & noise) const
    {
        assert(noise.mean().isZero());
        Gaussian out;
        Eigen::MatrixX<Scalar> C;
        out.mu_ = h(mu_, C) /*+ noise.mean()*/;
        const std::size_t & nx = mu_.rows();
        const std::size_t & ny = out.mu_.rows();
        Eigen::MatrixX<Scalar> SS(nx + ny, ny);
        SS << S_*C.transpose(), noise.sqrtCov();
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SS);   // In-place QR decomposition
        out.S_ = SS.topRows(ny).template triangularView<Eigen::Upper>();
        return out;
    }

    // log likelihood and derivatives
    Scalar log(const Eigen::VectorX<Scalar> & x) const
    {
        assert(x.cols() == 1);
        assert(x.size() == size());

        // Compute log N(x; mu, P) where P = S.'*S
        // log N(x; mu, P) = -0.5*(x - mu).'*inv(P)*(x - mu) - 0.5*log(det(2*pi*P))

        // TODO: Numerically stable version
        
        // Really bad version
        Eigen::MatrixX<Scalar> P = S_.transpose()*S_;   // Bad, because unnecessary and loss of precision
        Eigen::MatrixX<Scalar> Pinv = P.inverse();      // Bad, because you should know better (https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/)
        Scalar quadraticForm = (x - mu_).transpose()*Pinv*(x - mu_);
        using std::log, std::sqrt, std::exp;            // Bring selected math functions into global namespace, to merge with autodiff:: provided functions
        return log( 1.0/sqrt( (2*M_PI*P).determinant() )*exp(-0.5*quadraticForm) ); // Bad, because determinant, underflow, overflow and loss of precision
    }

    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g) const
    {
        // TODO: Compute gradient of log N(x; mu, P) w.r.t x and write it to g

        return log(x);
    }

    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g, Eigen::MatrixX<Scalar> & H) const
    {
        // TODO: Compute Hessian of log N(x; mu, P) w.r.t x and write it to H

        return log(x, g);
    }

    Gaussian operator*(const Gaussian & other) const
    {
        const std::size_t & n1 = size();
        const std::size_t & n2 = other.size();
        Gaussian out(n1 + n2);
        out.mu_ << mu_, other.mu_;
        out.S_ << S_,                                  Eigen::MatrixXd::Zero(n1, n2),
                Eigen::MatrixX<Scalar>::Zero(n2, n1), other.S_;
        return out;
    }

    Eigen::VectorX<Scalar> simulate() const
    {
        static boost::random::mt19937 rng(std::time(0));    // Initialise and seed once
        boost::random::normal_distribution<> dist;

        // Draw w ~ N(0, I)
        Eigen::VectorX<Scalar> w(size());
        for (Eigen::Index i = 0; i < size(); ++i)
        {
            w(i) = dist(rng);
        }

        return mu_ + S_.transpose()*w;
    }
protected:
    Eigen::VectorX<Scalar> mu_;
    Eigen::MatrixX<Scalar> S_;
};

#endif

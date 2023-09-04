#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <cstddef>
#include <cmath>
#include <ctime>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/Core>
#include <Eigen/QR>

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

        //std::cout << "Gaussian.hpp S_ = " << S_ << std::endl;
        //std::cout << "Gaussian.hpp Marginal mu= " << mu_ << std::endl;

        Gaussian out(idx.size());
        out.mu_ = mu_(idx);

        Eigen::MatrixXd temp = S_(Eigen::all, idx);

        Eigen::HouseholderQR<Eigen::MatrixXd> qr(temp);

        // Extract upper triangular matrix
        Eigen::MatrixXd qr_matrix = qr.matrixQR();

        
        if (qr_matrix.rows() != qr_matrix.cols()) {
            qr_matrix.conservativeResize(qr_matrix.cols(), qr_matrix.cols()); // Resize to square by removing rows
        }

        out.S_ = qr_matrix.triangularView<Eigen::Upper>();
        
        //std::cout << "Gaussian.hpp out.S_ = " << out.S_ << std::endl;
        //std::cout << "Gaussian.hpp Marginal out.mu_ = " << out.mu_ << std::endl;
        
        return out;
    }

    // Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
    template <typename IndexTypeA, typename IndexTypeB>
    Gaussian conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorX<Scalar> & xB) const
    {
        Gaussian out;

        // Get size of A and B indexes
        const int &nb = idxB.size();
        const int &na = idxA.size();
        
        // Make Horizontal concatenate matrix
        Eigen::MatrixXd S(S_.rows(), nb + na);

        // Concatenate indexed B and A horizontally
        S << S_(Eigen::all,idxB), S_(Eigen::all,idxA);

        // Perform QR decomposition
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(S);
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

        /*
            Where R = [ R1 R2 ]
                    [  0 R3 ]
            
            Hence,
                - R1 = R.topLeftCorner(nb, nb);
                - R2 = R.topRightCorner(nb, na);
                - R3 = R.bottomRightCorner(na, na); 

            Used for equation 31) Lab5

                mu_A/B = mu_A + R2.T*R1.-T*( xB - mu_B )

            and 

                S_A/B = R3 
        */

        // Plug and Chug maths - Note the solve -> 'Lower' used becuase R1 is transposed upper triangle
        out.mu_ = mu_(idxA) + 
                R.topRightCorner(nb, na).transpose()*
                (R.topLeftCorner(nb, nb).transpose()).triangularView<Eigen::Lower>().solve(xB - mu_(idxB));

        out.S_ = R.bottomRightCorner(na, na);
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
    Gaussian transform(Func h, const Gaussian & noise) const
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

        // From LAB 1 MCHA4100 'logGaussian.m' utilising the same notation
        Eigen::MatrixX<Scalar> w = S_.transpose().template triangularView<Eigen::Lower>().solve(x - mu_);
        
        // https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
        // "...(squaredNorm) is equal to the dot product of the vector by itself, and equivalently 
        // to the sum of squared absolute values of its coefficients"
        Scalar quadraticForm = w.squaredNorm();

        // https://au.mathworks.com/matlabcentral/fileexchange/22026-safe-computation-of-logarithm-determinat-of-large-matrix//
        // The above was used to get logdet out
        // Line from function library
        // v = 2 * sum(log(diag(chol(A))));
        // Note that the 2 multiple is ommitted because of the 0.5 in the final equation
        // Scalar logdet = S_.template diagonal().array().abs().log().sum(); 
        Scalar logdet = S_.diagonal().array().abs().log().sum(); 
 
        // Note that the determinant is log form which follows the log rule log(AB) = log(A) + log(B)
        // The following rule is also used log(m)^n = n*log(m)
        // Equation from Week 6 Lecture page 14
        Scalar loglikelihood = -0.5 * x.size() * std::log(2 * M_PI) - logdet - 0.5 * quadraticForm;

        return loglikelihood;
    }

    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g) const
    {
        // Compute gradient of log N(x; mu, P) w.r.t x and write it to g

        g = -S_.template triangularView<Eigen::Upper>().solve(S_.transpose().template triangularView<Eigen::Lower>().solve(x - mu_));

        return log(x);
    }

    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g, Eigen::MatrixX<Scalar> & H) const
    {
        // Compute Hessian of log N(x; mu, P) w.r.t x and write it to H

        H = -S_.template triangularView<Eigen::Upper>().solve(S_.transpose().template triangularView<Eigen::Lower>().solve(Eigen::MatrixX<Scalar>::Identity(S_.rows(), S_.cols())));

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

    // https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution
    static double chi2inv(double p, double nu)
    {
        assert(p >= 0);
        assert(p < 1);

        // Calculate the inverse of the incomplete gamma function to get the chi-square inverse
        double chi2_inv = 2 * boost::math::gamma_p_inv(nu / 2.0, p);

        return chi2_inv;
    }

    static double normcdf(double w)
    {
        // Calculate the cumulative distribution function (CDF) of the standard normal distribution
        double cdf = 0.5 * boost::math::erfc(-w / std::sqrt(2.0));

        return cdf;
    }

    bool isWithinConfidenceRegion(const Eigen::VectorX<Scalar> & x, double nSigma = 3.0)
    {
        const Eigen::Index & n = size();
        
        double c = 2 * normcdf(nSigma) - 1;
        double chi2_inv = chi2inv(c, n);

        // From LAB 1 MCHA4100 'logGaussian.m' utilising the same notation
        Eigen::MatrixX<Scalar> w = S_.transpose().template triangularView<Eigen::Lower>().solve(x - mu_);
        
        // https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
        // "...(squaredNorm) is equal to the dot product of the vector by itself, and equivalently 
        // to the sum of squared absolute values of its coefficients"
        Scalar quadraticForm = w.squaredNorm();

        if (quadraticForm <= chi2_inv){
            return true;
        }
        else{
            return false;
        }

        return false;
    }

    // Points on boundary of confidence ellipse for a given number of standard deviations
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> confidenceEllipse(double nSigma = 3.0, int nSamples = 100) const
    {
        const Eigen::Index & n = size();
        assert(n == 2);

        Eigen::Matrix<Scalar, 2, Eigen::Dynamic> X(2, nSamples);
        
        double c = 2 * normcdf(nSigma) - 1;
        Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(nSamples, 0.0, 2.0 * M_PI);

        double r = std::sqrt(chi2inv(c,n));

        Eigen::MatrixXd w(2, nSamples);

        for (int i = 0; i < nSamples; i++){
            w(0, i) = r * std::cos(t(i));  // Filling the first row with cos(t) values
            w(1, i) = r * std::sin(t(i));  // Filling the second row with sin(t) values
        }

        X = S_.transpose() * w;

        for (int i = 0; i < nSamples; i++){
            X(0, i) = mu_(0) + X(0, i);
            X(1, i) = mu_(1) + X(1, i);
        }

        assert(X.cols() == nSamples);
        assert(X.rows() == 2);
        return X;
    }

    // Quadric surface coefficients for a given number of standard deviations
    Eigen::Matrix4<Scalar> quadricSurface(double nSigma = 3.0) const
    {
        const Eigen::Index & n = size();
        assert(n == 3);
        
        Eigen::Matrix4<Scalar> Q;
        
        Q.topLeftCorner(3,3) = S_.template triangularView<Eigen::Upper>().solve(S_.transpose().template triangularView<Eigen::Lower>().solve(Eigen::MatrixX<Scalar>::Identity(S_.rows(), S_.cols())));
        //std::cout << "Gaussian.hpp TopLeft = " << Q.topLeftCorner(3,3) << std::endl;
        Eigen::MatrixX<Scalar> z;
        z = S_.transpose().template triangularView<Eigen::Lower>().solve(mu_);
        //std::cout << "Gaussian.hpp z = " << z << std::endl;

        Eigen::MatrixX<Scalar> y;
        y = S_.template triangularView<Eigen::Upper>().solve(z);
        //std::cout << "Gaussian.hpp y = " << y << std::endl;

        Q.topRightCorner(3, 1) = -y;
        Q.bottomLeftCorner(1, 3) = -y.transpose();

        double c = 2 * normcdf(nSigma) - 1;
        double chi2_inv = chi2inv(c, 3);

        Scalar z_Norm = z.squaredNorm();

        Q(3,3) = z_Norm - chi2_inv;

        return Q;
    }

protected:
    Eigen::VectorX<Scalar> mu_;
    Eigen::MatrixX<Scalar> S_;
};

#endif

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

    out.mu_ = mu_(idx);

    // Perform QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(S_(Eigen::all,idx));

    // Extract upper triangular matrix
    out.S_ = qr.matrixQR().triangularView<Eigen::Upper>();

    return out;
}

// Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
template <typename IndexTypeA, typename IndexTypeB>
Gaussian Gaussian::conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd & xB) const
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

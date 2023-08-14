// Tip: Only include headers needed to parse this implementation only
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>

#include "gaussian_util.h"

// TODO: Function implementations

void pythagoreanQR(const Eigen::MatrixXd &S1, const Eigen::MatrixXd &S2, Eigen::MatrixXd &S) {
    // Implementation of the pythagoreanQR function
    // ...

    // Stack S1 and S2 vertically
    Eigen::MatrixXd combinedMatrix(S1.rows() + S2.rows(), S1.cols()); // Create matrix the with correct dimensions for [S1 S2].T
    
    // Stack S1 and S2 into combined Matrix
    combinedMatrix << S1,
                      S2;

    // Perform QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(combinedMatrix);

    // Extract upper triangular matrix
    S = qr.matrixQR().triangularView<Eigen::Upper>();
}

void conditionGaussianOnMarginal(const Eigen::VectorXd & muyxjoint, const Eigen::MatrixXd & Syxjoint, const Eigen::VectorXd & y, Eigen::VectorXd & muxcond, Eigen::MatrixXd & Sxcond)
{
    // Implementation of the condition Gaussian on Marginal function
    // ...

    int ny = y.size();
    int nx = muyxjoint.size() - ny; // muyxjoint is size [nx + ny] array;

    // Extract Elements from Syxjoint
    Eigen::MatrixXd S1 = Syxjoint.block(0, 0, ny, ny);      // Top Left Matrix Block
    Eigen::MatrixXd S2 = Syxjoint.block(0, ny, ny, nx);     // Top Right Matrix Block
    Eigen::MatrixXd S3 = Syxjoint.block(ny, ny, nx, nx);    // Bottom Right Matrix Block -> Bottom Left Matrix Block is zeros

    // Decompose muyxjoint
    Eigen::VectorXd muy = muyxjoint.head(ny);
    Eigen::VectorXd mux = muyxjoint.tail(nx);

    // Solve the computational part - note 'Lower' -> because S1 is upper triangle and has been tranposed
    // template m.triangularView<Eigen::Upper>().solve(n)
    Eigen::MatrixXd S_solve = (S1.transpose()).triangularView<Eigen::Lower>().solve(y - muy);

    // Equation 8a - Plug and Chug Mathematics
    muxcond = mux + S2.transpose()*S_solve;

    // Equation 8b
    Sxcond = S3;
}
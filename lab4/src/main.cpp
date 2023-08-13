#include <cassert>
#include <cstdlib>
#include <iostream>
#include <Eigen/Core>

int main(int argc, char *argv[])
{
    std::cout << "Eigen version: ";
    // TODO a)
    std::cout << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "\n" << std::endl;

    std::cout << "Create a column vector:" << std::endl;
    Eigen::VectorXd x(3);
    // TODO b)
    x << 1, 3.2, 0.01; // Populate the vector using the << assignment operator
    std::cout << "x = \n" << x << "\n" << std::endl;

    std::cout << "Create a matrix:" << std::endl;
    Eigen::MatrixXd A;
    // TODO: c) Don't just use a for loop or hardcode all the elements
    //          Try and be creative :)

    // Create row and column vectors for broadcasting
    Eigen::VectorXd row_vector = Eigen::VectorXd::LinSpaced(4, 1, 4);
    Eigen::VectorXd col_vector = Eigen::VectorXd::LinSpaced(3, 1, 3);

    // Perform broadcasting to fill the matrix
    A = row_vector * col_vector.transpose();

    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "A.rows() = " << A.rows() << std::endl;
    std::cout << "A.cols() = " << A.cols() << std::endl;
    std::cout << "A = \n" << A << "\n" << std::endl;
    std::cout << "A.transpose() = \n" << A.transpose() << "\n" << std::endl;

    std::cout << "Matrix multiplication:" << std::endl;
    Eigen::VectorXd Ax;
    // TODO d)
    Ax = A * x;

    std::cout << "A*x = \n" << Ax << "\n" << std::endl;

    std::cout << "Matrix concatenation:" << std::endl;
    Eigen::MatrixXd B(A.rows(), 2 * A.cols());
    // TODO e)
    B << A, A;
    std::cout << "B = \n" << B << "\n" << std::endl;
    Eigen::MatrixXd C(2 * A.rows(), A.cols());
    // TODO e)
    C << A, A;
    std::cout << "C = \n" << C << "\n" << std::endl;

    std::cout << "Submatrix via block:" << std::endl;
    Eigen::MatrixXd D;
    // TODO f)
    D = B.block(1, 2, 1, 3);
    std::cout << "D = \n" << D << "\n" << std::endl;
    std::cout << "Submatrix via slicing:" << std::endl;
    // TODO f)
    D = B(Eigen::seq(1, 1), Eigen::seqN(2, 3));
    std::cout << "D = \n" << D << "\n" << std::endl;

    std::cout << "Broadcasting:" << std::endl;
    Eigen::VectorXd v(6);
    Eigen::MatrixXd E;
    // TODO g)
    v << 1, 3, 5, 7, 4, 6;
    E = B.rowwise() + v.transpose();
    std::cout << "E = \n" << E << "\n" << std::endl;

    std::cout << "Index subscripting:" << std::endl;
    Eigen::MatrixXd F;
    // TODO h)

    // Define the vectors r and c
    Eigen::VectorXi r(4);
    r << 1, 3, 2, 4;
    Eigen::VectorXi c(6);
    c << 1, 4, 2, 5, 3, 6;

    F = B(r.array() - 1, c.array() - 1);

    std::cout << "F = \n" << F << "\n" << std::endl;
    // TODO i)
    std::cout << "Memory mapping:" << std::endl;
    float array[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    //Eigen::Matrix3f G;              // TODO: Replace this with an Eigen::Map
    Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> G(array);
    array[2] = -3.0f;               // Change an element in the raw storage
    assert(array[2] == G(0,2));     // Ensure the change is reflected in the view
    G(2,0) = -7.0f;                 // Change an element via the view
    assert(G(2,0) == array[6]);     // Ensure the change is reflected in the raw storage
    std::cout << "G = \n" << G << "\n" << std::endl;
    std::cout << "Array = \n" << array[2] << "\n" << std::endl;

    return EXIT_SUCCESS;
}

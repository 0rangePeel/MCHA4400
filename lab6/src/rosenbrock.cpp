#include <Eigen/Core>
#include "rosenbrock.hpp"

// Functor for Rosenbrock function and its derivatives
double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x)
{
    return rosenbrock(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    g.resize(2, 1);
    // TODO: Write gradient to g

    return operator()(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    H.resize(2, 2);
    // TODO: Write Hessian to H

    return operator()(x, g);
}

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

    g << -2 + 2*x(0) - 400*x(1)*x(0) + 400*std::pow(x(0),3), 200*x(1) - 200*std::pow(x(0),2);

    return operator()(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    H.resize(2, 2);

    H <<800*pow(x(0),2) - 400*(x(1) - pow(x(0),2)) + 2   , -400*x(0),
        -400*x(0)                   ,   200;

    return operator()(x, g);
}

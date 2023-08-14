#include <cmath>
#include <Eigen/Core>
#include "Gaussian.h"
#include "State.h"
#include "StateBallistic.h"

const double StateBallistic::p0 = 101.325e3;            // Air pressure at sea level [Pa]
const double StateBallistic::M  = 0.0289644;            // Molar mass of dry air [kg/mol]
const double StateBallistic::R  = 8.31447;              // Gas constant [J/(mol.K)]
const double StateBallistic::L  = 0.0065;               // Temperature gradient [K/m]
const double StateBallistic::T0 = 288.15;               // Temperature at sea level [K]
const double StateBallistic::g  = 9.81;                 // Acceleration due to gravity [m/s^2]

StateBallistic::StateBallistic(const Gaussian & density)
    : State(density)
{
    // SQ_ is an upper triangular matrix such that SQ_.'*SQ_ = Q is the power spectral density of the continuous time process noise
    SQ_ << 0,     0,    0,
           0, 1e-10,    0,
           0,     0, 5e-6;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd StateBallistic::dynamics(const Eigen::VectorXd & x) const
{
    Eigen::VectorXd f(x.size());

    /*
    From Lab5 Equation 1)
    
        x = [ h v c ].T

    and 

    From Lab5 Equation 10)

        d = 0.5*((M*p0)/R)*(1/(T0 - L*h))*std::pow((1 - (L*h)/(T0)),((g*M)/(R*L)))*std::pow(v,2)*c;

    From Lab5 Equation 3)

        f = [ f1 f2 f3 ]

    Where:
        - f1 = v;
        - f2 = d - g;
        - f3 = 0;

    */

    double d = 0.5*((M*p0)/R)*(1/(T0 - L*x(0)))*std::pow((1 - (L*x(0))/(T0)),((g*M)/(R*L)))*std::pow(x(1),2)*x(2);

    f << x(1), d - g, 0;

    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd StateBallistic::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(x);

    J.resize(x.size(), x.size());

    /*
    Where:
        - j21 = df2/dh
        - j22 = df2/dv
        - j23 = df2/dc

    */

    double j21 = (L * M * x(2) * p0 * std::pow(x(1),2) * std::pow(1 - (L * x(0)) / T0, (M * g) / (L * R))) / (2 * R * std::pow(T0 - L * x(0), 2)) - 
                 (std::pow(M, 2) * x(2) * g * p0 * std::pow(x(1),2) * std::pow(1 - (L * x(0)) / T0, (M * g) / (L * R) - 1)) / (2 * std::pow(R,2) * T0 * (T0 - L * x(0)));
    double j22 = ((M*p0)/R)*(1/(T0 - L*x(0)))*std::pow((1 - (L*x(0))/(T0)),((g*M)/(R*L)))*x(1)*x(2); // Note the 0.5*2*x after derivative
    double j23 = 0.5*((M*p0)/R)*(1/(T0 - L*x(0)))*std::pow((1 - (L*x(0))/(T0)),((g*M)/(R*L)))*std::pow(x(1),2);

    J << 0,    1,   0,
        j21, j22, j23,
         0,    0,   0;

    return f;
}

#include <doctest/doctest.h>
#include <Eigen/Core>
#include <filesystem>
#include <vector>
#include "../../src/Gaussian.hpp"
#include "../../src/dataAssociation.h"

SCENARIO("Joint compatibility")
{
    GIVEN("1 landmark, 3 features, 1 compatible feature")
    {
        double                  surprisal;
        std::vector<int>        idx{1};
        double                  nSigma = 3.0;

        // muY - [2 x 1]: 
        Eigen::VectorXd muY(2); 
        muY <<        1658.780527,
                      963.3626597;

        // SYY - [2 x 2]: 
        Eigen::MatrixXd SYY(2, 2); 
        SYY <<       -53.41061755,       9.901075665,
                                0,       -37.3526274;

        Gaussian p(muY, SYY);

        // y - [2 x 3]: 
        Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, 3); 
        Y <<        1754.156364,       1714.701131,       1921.012306,
                    765.1206969,       934.6000308,       849.6044461;

        // Surprisal per unassociated landmark
        double sU   = 14.832478857591873;
                                
                                
        WHEN("Calling jointCompatibility")
        {
            bool cA             =  jointCompatibility(idx, sU, Y, p, nSigma, surprisal);

            THEN("Association is compatible")
            {
                CHECK(cA == true);
            }

            THEN("Surprisal is correct")
            {
                CHECK(surprisal == doctest::Approx( 10.105667572052072));
            }
        }
    }

    GIVEN("1 landmark, 3 features, 0 compatible features")
    {
        double                  surprisal;
        std::vector<int>        idx{-1};
        double                  nSigma = 3.0;

        // muY - [2 x 1]: 
        Eigen::VectorXd muY(2); 
        muY <<        1658.780527,
                      963.3626597;

        // SYY - [2 x 2]: 
        Eigen::MatrixXd SYY(2, 2); 
        SYY <<       -53.41061755,       9.901075665,
                                0,       -37.3526274;

        Gaussian p(muY, SYY);

        // y - [2 x 3]: 
        Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, 3); 
        Y <<        1754.156364,       1714.701131,       1921.012306,
                    765.1206969,       934.6000308,       849.6044461;

        // Surprisal per unassociated landmark
        double sU   = 14.832478857591873;
                                
                                
        WHEN("Calling jointCompatibility")
        {
            bool cA             =  jointCompatibility(idx, sU, Y, p, nSigma, surprisal);

            THEN("Vacuous truth. We have an association with the uniform null hypothesis")
            {
                CHECK(cA == true);
            }

            THEN("Surprisal is correct")
            {
                CHECK(surprisal == doctest::Approx( 14.832478857591873));
            }
        }
    }


    GIVEN("3 landmarks, 13 features, 3 compatible features")
    {
        double                  surprisal;
        std::vector<int>        idx{6, 7, 10};
        double                  nSigma = 3.0;

        // muY - [6 x 1]: 
        Eigen::VectorXd muY(6); 
        muY <<        1658.780527,
                      963.3626597,
                       433.112825,
                       731.914696,
                      1410.293147,
                      577.2375667;

        // SYY - [6 x 6]: 
        Eigen::MatrixXd SYY(6, 6); 
        SYY <<       -53.41061755,       9.901075665,       5.306940133,       7.303991992,       -5.72141011,       6.609727066,
                                0,       -37.3526274,     0.08127966678,      -22.40730348,     -0.2022231471,      -15.87273879,
                                0,                 0,       36.03250166,      -22.71362239,      -1.725509191,       3.766078776,
                                0,                 0,                 0,      -41.70917897,       2.529976157,      -8.843062797,
                                0,                 0,                 0,                 0,       22.34877992,      -22.53303115,
                                0,                 0,                 0,                 0,                 0,       32.74457988;

        Gaussian p(muY, SYY);

        // y - [2 x 13]: 
        Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, 13); 
        Y <<                549,               942,              1151,                30,       1381.755157,       1757.892465,       1637.791659,       443.3084837,       279.9869569,       359.2704718,       1363.489534,       1387.401497,       1387.120065,
                            855,               625,              1163,               454,       986.4901225,       835.5843074,       1008.082601,       654.6662577,       862.0112818,       747.6976779,       603.0677591,       692.7547721,       563.2515042;


        // Surprisal per unassociated landmark
        double sU   = 14.832478857591873;
                                
                                
        WHEN("Calling jointCompatibility")
        {
            bool cA             =  jointCompatibility(idx, sU, Y, p, nSigma, surprisal);

            THEN("Association is compatible")
            {
                CHECK(cA == true);
            }

            THEN("Surprisal is correct")
            {
                CHECK(surprisal == doctest::Approx( 33.394861266913537));
            }
        }
    }
}
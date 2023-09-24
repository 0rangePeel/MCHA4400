#ifndef CHECKFEATURES_H
#define CHECKFEATURES_H 

#include "Camera.h"
#include "imagefeatures.h"
#include <Eigen/Core>

struct checkFeatureResult {
    std::vector<int> ids;
    Eigen::VectorXd y;
};

checkFeatureResult checkfeature(const ArUcoResult &arucoResult, const Camera &cam); 

bool isPointInsideEllipse(int x, int y, const Camera &cam);

#endif
#ifndef STATE_H
#define STATE_H

#include <cstddef>
#include <Eigen/Core>
#include "Gaussian.hpp"

class State
{
public:
    Gaussian<double> density;
    virtual ~State();
    explicit State(std::size_t n);
    explicit State(const Gaussian<double> & density);

    std::size_t size() const;
    void predict(double time);
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x) const = 0;
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const = 0;

    // Getter functions
    std::vector<int> getIdsLandmarks() const;
    std::vector<std::size_t> getIdxLandmarks() const;
    std::vector<std::size_t> getIdsHistLandmarks() const;

    // Setter functions
    void setIdsLandmarks(const std::vector<int>& ids);
    void setIdxLandmarks(const std::vector<std::size_t>& idx);
    void setIdsHistLandmarks(const std::vector<std::size_t>& idsHist);

    // Function to modify idsLandmarks using a std::vector input
    void modifyIdsLandmarks(const std::vector<int>& newIds);
    // Function to modify idxLandmarks using a std::vector input
    void modifyIdxLandmarks(const std::vector<std::size_t>& newIdx);
    // Function to modify idsHistLandmarks using a std::vector input
    void modifyIdsHistLandmarks(const std::vector<std::size_t>& newIdsHist);


protected:
    Eigen::MatrixXd augmentedDynamics(const Eigen::MatrixXd & X) const;
    Eigen::VectorXd RK4SDEHelper(const Eigen::VectorXd & xdw, double dt, Eigen::MatrixXd & J) const;
    Eigen::MatrixXd SQ_;
private:
    double time_;
    std::vector<int> idsLandmarks;
    std::vector<std::size_t> idxLandmarks;
    std::vector<std::size_t> idsHistLandmarks;
};

#endif

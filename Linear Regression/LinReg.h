#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>

struct LinearRegression {

    LinearRegression() {}

    double OLS_loss(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, Eigen::VectorXd& theta);

    std::tuple<Eigen::VectorXd, std::vector<double>> GradDesc(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, Eigen::VectorXd& theta, const double& a, const int& k);

    double R2(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_hat);

};
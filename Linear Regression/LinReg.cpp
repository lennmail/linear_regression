#include "LinReg.h"
#include "ETL.h"

#include <cmath>
#include <fstream>


double LinearRegression::OLS_loss(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, Eigen::VectorXd& theta) {

    Eigen::MatrixXd ind_loss = pow(((X * theta) - y).array(), 2);

    return (ind_loss.sum() / (2 * X.rows()));
}

std::tuple<Eigen::VectorXd, std::vector<double>> LinearRegression::GradDesc(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, Eigen::VectorXd& theta, const double& a, const int& k) {

    Eigen::MatrixXd temp = theta;

    int param = theta.rows();

    std::vector<double> loss;
    loss.push_back(OLS_loss(X, y, theta));

    for (int i = 0; i < k; i++) {
        Eigen::MatrixXd error = X * theta - y;
        for (int j = 0; j < param; j++) {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd term = error.cwiseProduct(X_i);
            temp(j, 0) = theta(j, 0) - ((a / X.rows()) * term.sum());
        }
        theta = temp;
        loss.push_back(OLS_loss(X, y, theta));
    }

    return std::make_tuple(theta, loss);

}

double LinearRegression::R2(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_hat) {

    auto num = pow((y - y_hat).array(), 2).sum();
    auto den = pow(y.array() - y.mean(), 2).sum();

    return 1 - num / den;
}
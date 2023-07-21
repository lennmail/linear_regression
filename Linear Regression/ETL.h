#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

class ETL {

    std::string dataset;
    std::string delim;
    bool header;

public:

    ETL(const std::string& dataset, const std::string& delim, const bool& header) :
        dataset(dataset), delim(delim), header(header) {}

    std::vector<std::vector<std::string>> readCSV();

    Eigen::MatrixXd toMat(const std::vector<std::vector<std::string>>& dataset, int& rows, const int& cols);

    auto Mean(const Eigen::MatrixXd& data) -> decltype(data.colwise().mean());
    auto STD(const Eigen::MatrixXd& data) -> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt());

    Eigen::MatrixXd norm(Eigen::MatrixXd& data);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(const Eigen::MatrixXd& data, const float& size);

    void VecToFile(const std::vector<double>& vec, const std::string& filename);
    void MatToFile(const Eigen::MatrixXd& data, const std::string& filename);

};
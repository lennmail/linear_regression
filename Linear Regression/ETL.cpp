#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <fstream>

#include "ETL.h"

std::vector<std::vector<std::string>> ETL::readCSV() {

    // open file
    std::ifstream file(dataset);
    std::vector<std::vector<std::string>> dataS;

    std::string line = "";

    while (getline(file, line)) {     // parse every line of the data into the variable "line"
        std::vector<std::string> temp;
        boost::algorithm::split(temp, line, boost::is_any_of(delim));   //split the line at every deliminater
        dataS.push_back(temp);   // add single entries to dataframe
    }

    file.close();

    return dataS;
}

Eigen::MatrixXd ETL::toMat(const std::vector<std::vector<std::string>>& dataset, int& rows, const int& cols) {

    if (header) // get rid of headers
        rows -= 1;

    Eigen::MatrixXd myMatrix(cols, rows);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            myMatrix(j, i) = atof(dataset[i][j].c_str());   // fill matrix with data

    return myMatrix.transpose();    // transpose because columns and rows are switched
}

auto ETL::Mean(const Eigen::MatrixXd& data) -> decltype(data.colwise().mean()) {
    return data.colwise().mean();
}

auto ETL::STD(const Eigen::MatrixXd& data) -> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt()) {
    return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt(); //compute standard deviation
}

Eigen::MatrixXd ETL::norm(Eigen::MatrixXd& data) {

    // function to normalize data according to z-normalisation

    auto mean = Mean(data);
    Eigen::MatrixXd scaled = data.rowwise() - mean;
    auto std = STD(scaled);

    return (scaled.array().rowwise() / std);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ETL::TrainTestSplit(const Eigen::MatrixXd& data, const float& size) {

    int rows = data.rows();
    int train_rows = round(size);
    int test_rows = rows - size;

    // target var needs to be on the far right of the csv file!
    // Otherwise this code needs to be adjusted

    Eigen::MatrixXd train = data.topRows(train_rows);
    Eigen::MatrixXd X_train = train.leftCols(data.cols() - 1);
    Eigen::MatrixXd y_train = train.rightCols(1);           // split the data into training and testing

    Eigen::MatrixXd test = data.bottomRows(test_rows);
    Eigen::MatrixXd X_test = test.leftCols(data.cols() - 1);
    Eigen::MatrixXd y_test = test.rightCols(1);

    return std::make_tuple(X_train, y_train, X_test, y_test);
}

void ETL::VecToFile(const std::vector<double>& vec, const std::string& filename) {

    std::ofstream output_file(filename);
    std::ostream_iterator<double> output_iterator(output_file, "\n");
    std::copy(vec.begin(), vec.end(), output_iterator);

}

void ETL::MatToFile(const Eigen::MatrixXd& data, const std::string& filename) {

    std::ofstream output_file(filename);
    if (output_file.is_open())
        output_file << data << "\n";
}
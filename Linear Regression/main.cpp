/* This program run a linear regression on the input data. For this script to work, you need the ETS I wrote.
It can be found here: https://github.com/lennmail/ETL. Additionally, dimensions of the data have to be adjusted 
with every different dataset.*/

#include "LinReg.h"
#include "ETL.h"

using namespace std;

// select parameters
const double TRAIN_SIZE = 0.7;
const double LEARNING_RATE = 0.001;
const int EPOCHS = 500;

// select target directory
const string EXPORT_PATH_LOSS = "";
const string EXPORT_PATH_PARAM = "";
const string EXPORT_PATH_YHAT = "";

// data dimensions
const int DATA_ROWS = 10000;
const int DATA_COLS = 10;

int main(int argc, char* argv[]) {

    if (argc < 4) // print error if wrong usage
        cout << "Error: Insufficient arguments" << endl;

    ETL myETL(argv[1], argv[2], argv[3]);
    LinearRegression lm;

    vector<vector<string>> dataset = myETL.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataM = myETL.toMat(dataset, rows, cols);

    Eigen::MatrixXd dataNormed = myETL.norm(dataM);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> dataSplit = myETL.TrainTestSplit(dataNormed, TRAIN_SIZE);
    tie(X_train, y_train, X_test, y_test) = dataSplit;

    // need this for the intercept (coefficient of x0, and x0 is always 1)
    Eigen::VectorXd vec_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vec_test = Eigen::VectorXd::Ones(X_test.rows());

    // resize dataframes to account for the new vector
    X_train.conservativeResize(X_train.rows(), X_train.cols() + 1);
    X_train.col(X_train.cols() - 1) = vec_train;

    X_test.conservativeResize(X_test.rows(), X_test.cols() + 1);
    X_test.col(X_test.cols() - 1) = vec_test;

    // initialize parameters
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());

    Eigen::VectorXd theta_Final;
    vector<double> loss;

    tuple<Eigen::VectorXd, vector<double>> grad_desc = lm.GradDesc(X_train, y_train, theta, LEARNING_RATE, EPOCHS);
    tie(theta_Final, loss) = grad_desc;

    cout << theta_Final << endl;

    for (auto v : loss)
        cout << v << endl;

    // uncomment for export

    // myETL.VecToFile(loss, EXPORT_PATH_LOSS);
    // myETL.MatToFile(theta_Final, EXPORT_PATH_PARAM);

    // adjust number of columns and rows
    auto mu_data = myETL.Mean(dataM);
    auto mu_z = mu_data(0, DATA_COLS);
    auto scaled_data = dataM.rowwise() - dataM.colwise().mean();
    auto sigma_data = myETL.STD(scaled_data);
    auto sigma_z = sigma_data(0, DATA_COLS);

    Eigen::MatrixXd y_train_hat = (X_train * theta_Final * sigma_z).array() + mu_z;
    Eigen::MatrixXd y = dataM.col(DATA_COLS).topRows(DATA_ROWS);

    double R_Squared = lm.R2(y, y_train_hat);
    cout << "R2: " << R_Squared << endl;

    // myETL.MatToFile(y_train_hat, EXPORT_PATH_YHAT);

    return 0;
}
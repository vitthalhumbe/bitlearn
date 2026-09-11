#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

class LinearRegression
{
public:
    LinearRegression() {};

    void fit(py::array_t<double> X, py::array_t<double> y,string method,string regul,double lr,int epochs, double lambda_, double alpha) {
        auto Xbuffer = X.unchecked<2>();
        auto ybuffer = y.unchecked<1>();

        int m = Xbuffer.shape(0);       // rows
        int n = Xbuffer.shape(1) + 1;   // cols, +1 for bias

        // adding intercept

        Eigen::MatrixXd Xb(m,n);
        Eigen::VectorXd yv(m);
        for (int i = 0;i < m; ++i) {
            Xb(i, 0) = 1.0;
            for (int j = 1; j < n; ++j) {
                Xb(i, j) = Xbuffer(i, j-1);
            }
            yv(i) = ybuffer(i);
        }

        weights = Eigen::VectorXd::Zero(n);

        if (method == "normal_eq")
            fit_normal_eq(Xb, yv, regul, lambda_);
        else if (method == "gd") 
            fit_gd(Xb, yv, method, regul, lr, epochs, lambda_, alpha);
        else if (method == "sgd")
            fit_sgd(Xb, yv, method, regul, lr, epochs, lambda_, alpha);
        else
            throw runtime_error("Unknown method");
    }

    py::array_t<double> predict(py::array_t<double> X) { 
        auto Xbuf = X.unchecked<2>(); 
        int m = Xbuf.shape(0); 
        int n = weights.size(); 
        
        Eigen::MatrixXd Xb(m, n); 
        
        for (int i = 0; i < m; ++i) { 
            Xb(i, 0) = 1.0;
            for (int j = 1; j < n; ++j) 
                Xb(i, j) = Xbuf(i, j - 1); 
            
        } 
        Eigen::VectorXd y_pred = Xb * weights; 
        
        py::array_t<double> out(m); 
        auto outbuf = out.mutable_unchecked<1>(); 
        for (int i = 0; i < m; ++i) 
            outbuf(i) = y_pred(i); 
        return out;
    }
    py::array_t<double> get_weights() const {
        py::array_t<double> out(weights.size());
        auto buf = out.mutable_unchecked<1>();
        for (int i = 0; i < weights.size(); i++) {
            buf(i) = weights(i);
        }
        return out;
    };

    py::array_t<double> get_loss_history() const {
    py::array_t<double> out(loss_history.size());
    auto buf = out.mutable_unchecked<1>();

    for (int i = 0; i < loss_history.size(); ++i)
        buf(i) = loss_history[i];

    return out;
}

private:
    string method, regul;
    double lr, lambda_, alpha;
    int epochs;
    Eigen::VectorXd weights;
    std::vector<double> loss_history;

    // double dot();
    void add_regularization(Eigen::VectorXd& grad, const string& regul, double lambda_, double alpha) {
        for (int j = 1;j < grad.size(); ++j) {
            if (regul == "l2") {
                grad(j) += 2 * lambda_ * weights(j);
            } else if (regul == "l1") {
                grad(j) +=lambda_ * (weights(j) > 0 ? 1 : -1);
            } else if( regul == "elastic_net") {
                grad(j) += lambda_ * (alpha * (weights(j) > 0 ? 1: -1) + 2 * (1-alpha) * weights(j));
            }
        }
    }
    void fit_gd(const Eigen::MatrixXd& X, const Eigen::VectorXd& y ,string method,string regul,double lr,int epochs, double lambda_, double alpha) {
        int m = X.rows();
        loss_history.clear();

        for (int e = 0; e < epochs; e++) {
            Eigen::VectorXd error = X * weights -y;
            Eigen::VectorXd grad = (2.0 / m) * X.transpose() * error;

            add_regularization(grad, regul, lambda_, alpha);
            weights -= lr * grad;
            double loss = compute_loss(X, y, method, regul, lr, epochs,  lambda_, alpha);
            loss_history.push_back(loss);
        }
    }
    void fit_sgd(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,string method,string regul,double lr,int epochs, double lambda_, double alpha) {
        int m = X.rows();
         loss_history.clear();


        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < m; i++) {
                double err = X.row(i).dot(weights) - y(i);
                Eigen::VectorXd grad = 2 * err * X.row(i).transpose();
                
                add_regularization(grad, regul, lambda_, alpha);
                weights -= lr * grad;
            }
            double loss = compute_loss(X, y, method, regul, lr, epochs,  lambda_, alpha);
        loss_history.push_back(loss);
        }
    };
    void fit_normal_eq(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, string regul, double lambda_){
        Eigen::MatrixXd XtX = X.transpose() * X;
        if (regul == "l2") {
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
            I(0, 0) = 0.0;
            XtX += lambda_ * I;
        }
        weights = XtX.ldlt().solve(X.transpose() * y);
    };

    double compute_loss(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,string method,string regul,double lr,int epochs, double lambda_, double alpha) {
    Eigen::VectorXd error = X * weights - y;
    double mse = error.squaredNorm() / y.size();

    double reg = 0.0;
    for (int j = 1; j < weights.size(); ++j) {
        if (regul == "l2") {
            reg += lambda_ * weights(j) * weights(j);
        } else if (regul == "l1") {
            reg += lambda_ * std::abs(weights(j));
        } else if (regul == "elastic_net") {
            reg += lambda_ * (
                alpha * weights(j) * weights(j)
                + (1 - alpha) * std::abs(weights(j))
            );
        }
    }

    return mse + reg;
}

};

PYBIND11_MODULE(_linreg_cpp, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit)
        .def("predict", &LinearRegression::predict)
        .def("get_loss_history", &LinearRegression::get_loss_history)
        .def("get_weights", &LinearRegression::get_weights);
} 
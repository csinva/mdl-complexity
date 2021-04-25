//
//  glasso.cpp
//
//  Python bindings of graphical lasso
//
//  Created by Kohei Miyaguchi on 2017/06/10.
//  Copyright © 2017年 Kohei Miyaguchi. All rights reserved.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;


struct GraphicalLassoResult {
    Eigen::MatrixXd Theta, Sigma;
    bool converged;
    double gap;
    GraphicalLassoResult(long m) : Theta{m, m}, Sigma{Eigen::MatrixXd::Identity(m, m)}, converged{false}, gap{-1} {}
};


using GraphicalLasso = std::function<GraphicalLassoResult&(Eigen::MatrixXd const &, Eigen::MatrixXd const &)>;


double linesearch(const Eigen::MatrixXd& S, const Eigen::MatrixXd& W, const Eigen::MatrixXd& G, const Eigen::MatrixXd& Lambda, int iter_max=INT_MAX, bool verbose=false, double eps=1e-3);


GraphicalLasso graphicalLasso_stateful(long m, double tol, long iter_max, bool verbose, double eps) {
    using namespace Eigen;
    static const auto Zmm = MatrixXd::Zero(m, m);
    static const auto Ones = MatrixXd::Ones(m, m);
    MatrixXd W{m, m}, G{m, m};
    GraphicalLassoResult result(m);

    return [=](MatrixXd const &S, MatrixXd const &Lambda) mutable -> GraphicalLassoResult& {
        MatrixXd &Theta = result.Theta, &Sigma = result.Sigma;
        // Find feasible point of W assuming S and Sigma is SPD
        W = Sigma - S;
        MatrixXd shrinks = Lambda.array() / (W.cwiseAbs().array() + eps);
        double shrink = (Lambda.array() >= W.array()).select(shrinks, Ones).minCoeff();
        if (shrink < 1) {
            if (verbose) {
                printf("shrink rate: %f\n", shrink);
            }
            W *= shrink;
        }
        W.diagonal() = Lambda.diagonal();

        // Perform projected subgradient method over (S, Lambda, W, Theta) (not Sigma!)
        Theta = (S + W).inverse();
        double t = 0.0, gap = 2 * tol;
        for (int i = 0; i < iter_max; i++) {
            G = Theta;
            G.diagonal().setZero();
            G = (
                 ((W.array() >= Lambda.array()) * (G.array() > Zmm.array())) +
                 ((W.array() <= -Lambda.array()) * (G.array() < Zmm.array()))
                 ).select(Zmm, G);
            t = linesearch(S, W, G, Lambda, verbose);
            W = (W + t * G).cwiseMin(Lambda).cwiseMax(-Lambda);
            Theta = (S + W).inverse();
            S.cwiseProduct(Theta).sum();
            gap = S.cwiseProduct(Theta).sum() + Theta.cwiseAbs().cwiseProduct(Lambda).sum() - m;
            if (verbose) {
                printf("(glasso) step: %d, gap: %f\n", i, gap);
            }
            if (std::abs(gap / m) < tol) {
                result.converged = true;
                goto CONVERGED;
            }
        }
        result.converged = false;

    CONVERGED: // finish
        Theta = W.cwiseAbs().cwiseEqual(Lambda).select(Theta, Zmm);
        Sigma = S + W;
        result.gap = gap;
        return result;
    };
}


GraphicalLassoResult graphicalLasso(
        Eigen::MatrixXd const &S, Eigen::MatrixXd const &Lambda, Eigen::MatrixXd const &Sigma_init, Eigen::MatrixXd const &Theta_init,
        double tol, long iter_max, bool verbose, double eps) {
    using namespace Eigen;
    long m = S.rows();
    MatrixXd Zmm = MatrixXd::Zero(m, m);
    MatrixXd Ones = MatrixXd::Ones(m, m);
    MatrixXd W{m, m}, G{m, m};
    GraphicalLassoResult result(m);
    result.Theta = Theta_init;
    result.Sigma = Sigma_init;
    MatrixXd &Theta = result.Theta, &Sigma = result.Sigma;

    // Find feasible point of W assuming S and Sigma is SPD
    W = Sigma - S;
    MatrixXd shrinks = Lambda.array() / (W.cwiseAbs().array() + eps);
    double shrink = (Lambda.array() >= W.array()).select(shrinks, Ones).minCoeff();
    if (shrink < 1) {
        if (verbose) {
            printf("shrink rate: %f\n", shrink);
        }
        W *= shrink;
    }
    W.diagonal() = Lambda.diagonal();

    // Perform projected subgradient method over (S, Lambda, W, Theta) (not Sigma!)
    Theta = (S + W).inverse();
    double t = 0.0, gap = 2 * tol;
    for (int i = 0; i < iter_max; i++) {
        G = Theta;
        G.diagonal().setZero();
        G = (
             ((W.array() >= Lambda.array()) * (G.array() > Zmm.array())) +
             ((W.array() <= -Lambda.array()) * (G.array() < Zmm.array()))
             ).select(Zmm, G);

        t = linesearch(S, W, G, Lambda, verbose);
        W = (W + t * G).cwiseMin(Lambda).cwiseMax(-Lambda);
        Theta = (S + W).inverse();
        S.cwiseProduct(Theta).sum();

        gap = S.cwiseProduct(Theta).sum() + Theta.cwiseAbs().cwiseProduct(Lambda).sum() - m;
        if (verbose) {
            printf("(glasso) step: %d, gap: %f\n", i, gap);
        }

        if (std::abs(gap / m) < tol) {
            result.converged = true;
            goto CONVERGED;
        }
    }
    result.converged = false;

CONVERGED: // finish
    Theta = W.cwiseAbs().cwiseEqual(Lambda).select(Theta, Zmm);
    Sigma = S + W;

    Theta = 0.5 * (Theta + Theta.transpose());
    Sigma = 0.5 * (Sigma + Sigma.transpose());
    result.gap = gap;
    return result;
}


void test_glasso() {
    using namespace Eigen;

    MatrixXd S(3, 3), K(3, 3);

    K << 3.0, 1.0, 0.0,
    1.0, 2.0, 0.5,
    0.0, 0.5, 1.0;
    S = K.inverse();

    MatrixXd Lambda = MatrixXd::Ones(3, 3) * 0.1;
    MatrixXd I = MatrixXd::Identity(3, 3);

    printf("start testing glasso:\n");
    auto result = graphicalLasso(S, Lambda, I, I, 1e-10, 100, true, 1e-5);

    MatrixXd Thetatrue(3, 3), Sigmatrue(3, 3);
    Thetatrue <<   2.04478, 0.343284,        0,
    0.343284,   1.3808, 0.262195,
    0, 0.262195, 0.835366;
    Sigmatrue <<   0.511765, -0.135294, 0.0424646,
    -0.135294,  0.805882, -0.252941,
    0.0424646, -0.252941,   1.27647;

    std::cout << "Thetaguess\n" << result.Theta << std::endl;
    std::cout << "Thetatrue\n" << Thetatrue << std::endl;
    std::cout << "Sigmaguess\n" << result.Sigma << std::endl;
    std::cout << "Sigmatrue\n" << Sigmatrue << std::endl;

    double abserror = (result.Theta - Thetatrue).cwiseAbs().sum() + (result.Sigma - Sigmatrue).cwiseAbs().sum();
    printf("absolute error: %f\n", abserror);
    assert(0.001 > abserror);
}


// subroutine of glasso
double linesearch(const Eigen::MatrixXd& S, const Eigen::MatrixXd& W, const Eigen::MatrixXd& G, const Eigen::MatrixXd& Lambda, int iter_max, bool verbose, double eps) {
    using namespace Eigen;
    double f0 = log((S + W).determinant());
    MatrixXd SWG = (S + W).inverse() * G;

    double nom = SWG.diagonal().sum();
    double denom = (SWG.array() * SWG.transpose().array()).sum();
    if (verbose) {
        printf("(linesearch) step:-, nom:%f denom:%f, f0:%f\n", nom, denom, f0);
    }
    if (std::abs(denom) <= 0.0) return 0.0;
    double t = nom / denom;
    if (t <= 0) return 0.0;

    for (int i = 0; 0.0 < t && i < iter_max; i++) {
        double f = log((S + (W + t * G).cwiseMin(Lambda).cwiseMax(-Lambda)).determinant());
        if (verbose) {
            printf("(linesearch) step:%d, t:%f f:%f, f0:%f\n", i, t, f, f0);
        }
        if (f >= f0) break;
        t *= 0.5;
    }
    return t;
}


PYBIND11_PLUGIN(glassobind) {
    py::module m("glassobind", "graphical lasso plugin");
    py::class_<GraphicalLassoResult>(m, "GraphicalLassoResult")
        .def(py::init<long>())
        .def_readonly("theta", &GraphicalLassoResult::Theta)
        .def_readonly("sigma", &GraphicalLassoResult::Sigma)
        .def_readonly("converged", &GraphicalLassoResult::converged)
        .def_readonly("gap", &GraphicalLassoResult::gap);
    m.def("glasso", &graphicalLasso, "performs graphical lasso",
          "emp_cov"_a, "lambda"_a, "sigma_init"_a, "theta_init"_a, "tol"_a, "iter_max"_a, "verbose"_a, "eps"_a);
    m.def("test_glasso", &test_glasso, "test function");
    return m.ptr();
}

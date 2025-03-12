#pragma once

#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

Matrix<double, 5, 2> gaussQuad5() {
  VectorXd points = VectorXd::Zero(5);
  VectorXd weights = VectorXd::Zero(5);
  
  points(0) = - std::sqrt(5.0 + 2.0 * std::sqrt(10.0/7.0)) / 3.0;
  points(1) = - std::sqrt(5.0 - 2.0 * std::sqrt(10.0/7.0)) / 3.0;
  points(3) = - points(1);
  points(4) = - points(0);

  weights(0) = (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
  weights(1) = (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
  weights(2) = 128.0/225.0;
  weights(3) = weights(1);
  weights(4) = weights(0);

  // Make a matrix with two cols:
  // Column 0 are the points
  // Column 1 are the weights
  Matrix<double, 5, 2> gquad;
  gquad.col(0) = points;
  gquad.col(1) = weights;

  return gquad;
}

MatrixXd shiftedGaussQuad5(double xmin, double xmax, int N) {
  // Initialize the Gaussian weights and points on the
  // BOUNDED interval [-1, 1].
  Matrix<double, 5, 2> bgquad = gaussQuad5();

  // Break the integration domain [xmin, xmax] into N intervals
  double dx = (xmax - xmin) / N;

  VectorXd points = VectorXd::Ones(5*N);
  VectorXd weights = VectorXd::Ones(5*N);
  VectorXd range = VectorXd::LinSpaced(N+1, 0, N);

  VectorXd x = xmin + range.array() * dx;

  VectorXd half_diff = (x.tail(N) - x.head(N)) / 2;
  VectorXd midpoints = (x.tail(N) + x.head(N)) / 2;
  
  for (int k = 0; k < N; k++) {
    points( seq(5*k, 5*(k+1) - 1) ) = half_diff(k) * bgquad.col(0).array() + midpoints(k);
    weights( seq(5*k, 5*(k+1) - 1) ) = half_diff(k) * bgquad.col(1).array();
  }
  
  MatrixXd gquad = MatrixXd::Zero(5*N, 2);
  gquad.col(0) = points;
  gquad.col(1) = weights;
  
  return gquad;
}

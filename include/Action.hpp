#pragma once

#include <array>
#include <cmath>
#include "Sequence.hpp"

//constexpr double actionInterval = 4 * M_PI;
constexpr size_t numAction = 6;


VectorXd kappa(const VectorXd& centerTimes, const double harmonic, const double eta)
{
  return centerTimes.array() + eta * (harmonic * M_PI * centerTimes.array() / maxTime).sin();
}

/*
constexpr double eta1 = 0.04;//0.2;
constexpr double eta2 = 0.02;//0.1;
constexpr double eta3 = 0.0053;//0.1;
constexpr double eta4 = 0.01;//0.05;
constexpr double eta8 = 0.005;//0.05;

// Make typedef named CollectiveTransform which is a function that takes as argument VectorXd and returns VectorXd.
typedef VectorXd (*CollectiveTransform)(const VectorXd&);

// Do actionInterval and wbTimes method but with new functions and T = 1?

VectorXd reverseTransform(const VectorXd& centerTimes)
{
  return (maxTime - centerTimes.array()).reverse();
}

VectorXd kappa0(const VectorXd& centerTimes)
{
  return centerTimes.array();
}

VectorXd kappaPlus1(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta1 * (1.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaMinus1(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta1 * (-1.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaPlus2(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta2 * (2.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaMinus2(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta2 * (-2.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaPlus3(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta3 * (3.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaMinus3(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta3 * (-3.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaPlus4(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta4 * (4.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaMinus4(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta4 * (-4.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaPlus8(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta8 * (8.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaMinus8(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta8 * (-8.0 * M_PI * centerTimes.array() / maxTime).sin();
}

// MAKE SURE THE LENGTH OF THIS ARRAY IS EQUAL TO numAction !!!
const std::array<CollectiveTransform, numAction> actionList {
  kappaPlus1,
  kappaMinus1,
  kappaPlus4,
  kappaMinus4,
  kappaPlus8,
  kappaMinus8};
*/

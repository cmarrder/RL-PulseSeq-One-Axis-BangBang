#pragma once

#include <array>
#include <cmath>
#include "Sequence.hpp"

constexpr double actionInterval = 4 * M_PI;
constexpr size_t numAction = 7;
constexpr double eta1 = 0.2;
constexpr double eta2 = 0.1;
constexpr double eta4 = 0.05;

// Make typedef named CollectiveTransform which is a function that takes as argument VectorXd and returns VectorXd.
typedef VectorXd (*CollectiveTransform)(const VectorXd&);

// Do actionInterval and wbTimes method but with new functions and T = 1?

VectorXd reverseTransform(const VectorXd& centerTimes)
{
  return (maxTime - centerTimes.array()).reverse();
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

VectorXd kappaPlus4(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta4 * (4.0 * M_PI * centerTimes.array() / maxTime).sin();
}

VectorXd kappaMinus4(const VectorXd& centerTimes)
{
  return centerTimes.array() + eta4 * (-4.0 * M_PI * centerTimes.array() / maxTime).sin();
}

// MAKE SURE THE LENGTH OF THIS ARRAY IS EQUAL TO numAction !!!
const std::array<CollectiveTransform, numAction> actionList {
  kappaPlus1,
  kappaMinus1,
  kappaPlus2,
  kappaMinus2,
  kappaPlus4,
  kappaMinus4,
  reverseTransform};

/*
VectorXd sinHarmonic1Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + eta1 * (wbTimes.array() * 1.0 / 4.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic1Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - eta1 * (wbTimes.array() * 1.0 / 4.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic2Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + eta2 * (wbTimes.array() * 2.0 / 4.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic2Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - eta2 * (wbTimes.array() * 2.0 / 4.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic4Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + eta4 * (wbTimes.array() * 4.0 / 4.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic4Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - eta4 * (wbTimes.array() * 4.0 / 4.0).sin()) / actionInterval * maxTime;
}

// MAKE SURE THE LENGTH OF THIS ARRAY IS EQUAL TO numAction !!!
const std::array<CollectiveTransform, numAction> actionList {
  sinHarmonic1Plus,
  sinHarmonic1Minus,
  sinHarmonic2Plus,
  sinHarmonic2Minus,
  sinHarmonic4Plus,
  sinHarmonic4Minus,
  reverseTransform};
*/

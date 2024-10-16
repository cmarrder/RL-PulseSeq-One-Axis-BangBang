#pragma once

#include <array>
#include <cmath>
#include "Sequence.hpp"

constexpr double actionInterval = 4 * M_PI;
constexpr size_t numAction = 7;
constexpr double eta1 = 0.2;
constexpr double eta2 = 0.1;
constexpr double eta4 = 0.05;

// Make typedef named CollectiveTransform which is a function that takes as argument Sequence and returns void.
typedef VectorXd (*CollectiveTransform)(const VectorXd&);

VectorXd reverseTransform(const VectorXd& centerTimes)
{
  return (maxTime - centerTimes.array()).reverse();
}

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
  reverseTransform,
  sinHarmonic1Plus,
  sinHarmonic1Minus,
  sinHarmonic2Plus,
  sinHarmonic2Minus,
  sinHarmonic4Plus,
  sinHarmonic4Minus};


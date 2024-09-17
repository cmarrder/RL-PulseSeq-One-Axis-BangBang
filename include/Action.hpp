#pragma once

#include <array>
#include <cmath>
#include "Sequence.hpp"

constexpr double actionInterval = 2 * M_PI * 2;
constexpr size_t numAction = 9;
constexpr double phi = 0.4;

// Make typedef named CollectiveTransform which is a function that takes as argument Sequence and returns void.
typedef VectorXd (*CollectiveTransform)(const VectorXd&);

VectorXd reverseTransform(const VectorXd& centerTimes)
{
  return (maxTime - centerTimes.array()).reverse();
}

VectorXd sinHarmonic1Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + phi * (wbTimes.array() * 1.0 / 2.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic1Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - phi * (wbTimes.array() * 1.0 / 2.0).sin()) / actionInterval * maxTime;
}

VectorXd cosHarmonic1Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + phi * (wbTimes.array() * 1.0 / 2.0).cos() - phi) / actionInterval * maxTime;
}

VectorXd cosHarmonic1Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - phi * (wbTimes.array() * 1.0 / 2.0).cos() + phi) / actionInterval * maxTime;
}

VectorXd sinHarmonic2Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + phi * (wbTimes.array() * 2.0 / 2.0).sin()) / actionInterval * maxTime;
}

VectorXd sinHarmonic2Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - phi * (wbTimes.array() * 2.0 / 2.0).sin()) / actionInterval * maxTime;
}

VectorXd cosHarmonic2Plus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() + phi * (wbTimes.array() * 2.0 / 2.0).cos() - phi) / actionInterval * maxTime;
}

VectorXd cosHarmonic2Minus(const VectorXd& centerTimes)
{
  VectorXd wbTimes = centerTimes / maxTime * actionInterval;
  return (wbTimes.array() - phi * (wbTimes.array() * 2.0 / 2.0).cos() + phi) / actionInterval * maxTime;
}


// MAKE SURE THE LENGTH OF THIS ARRAY IS EQUAL TO numAction !!!
const std::array<CollectiveTransform, numAction> actionList {
  reverseTransform,
  sinHarmonic1Plus,
  sinHarmonic1Minus,
  cosHarmonic1Plus,
  cosHarmonic1Minus,
  sinHarmonic2Plus,
  sinHarmonic2Minus,
  cosHarmonic2Plus,
  cosHarmonic2Minus};


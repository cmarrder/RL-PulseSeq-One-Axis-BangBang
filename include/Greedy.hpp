#pragma once

#include <cmath>

double epsilonByProportion(const double p)
{
  static constexpr double epsilonStart = 1.0;
  static constexpr double epsilonFinal = 0.1;
  static constexpr double epsilonDecay = 0.3;

  return epsilonFinal + (epsilonStart - epsilonFinal) 
    * exp(-p / epsilonDecay);
}

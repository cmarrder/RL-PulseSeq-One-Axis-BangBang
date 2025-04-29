#pragma once

#include <cmath>

double epsilonByProportion(const double p)
{
  static constexpr double epsilonStart = 1.0;
  static constexpr double epsilonFinal = 0.1;//0.175;//0.1;
  static constexpr double epsilonDecay = 0.3;

  static constexpr double power = 1.25;

  //return epsilonFinal + (epsilonStart - epsilonFinal) 
  //  * exp(-p / epsilonDecay);
  return epsilonFinal + (epsilonStart - epsilonFinal) 
    * exp(-std::pow(p, power) / epsilonDecay);
}

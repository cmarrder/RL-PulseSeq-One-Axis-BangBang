#pragma once

#include <cmath>

#include "Crystal.hpp"
#include "Action.hpp"

const double SMALL = 1e-8;

constexpr size_t maxSteps = 10;

class Environment {

  Crystal crystal;
  std::vector<int> actionsDone;

public:

  void reset()
  {
    crystal.reset();
    actionsDone.clear();

  }

  //Environment() { reset(); }
  Environment(Param& param) : crystal(param) {} 

  Feature state() const
  {
    return crystal.feature();
  }

  size_t stepCount() const
  {
    return actionsDone.size();
  }

  bool done() const
  {
    return stepCount() >= maxSteps;
  }

  double reward() const
  {
   if (done()) {
      double chi = crystal.chi();
      return (fabs(chi) < SMALL ? 1 / SMALL : 1 / chi);
    } else
      return 0;
  }

  void applyAction(int action)
  {
    crystal.step(action);
    actionsDone.push_back(action);
  }

  std::vector<int> actionRecord() const
  {
    return actionsDone;
  }
};
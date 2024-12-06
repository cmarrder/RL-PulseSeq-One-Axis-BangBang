#pragma once

#include <cmath>
#include "Crystal.hpp"
#include "Action.hpp"

//const double SMALL = 1e-8;
//const double rewardScale = 1;//0.1;

constexpr size_t maxSteps = 20;

class Environment {

  Crystal crystal;
  std::vector<int> actionsDone;
  int rewardCalls = 0;

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

  //double reward() const
  double reward()
  {
    if (done())
    {
     //return crystal.getInitialChi() / crystal.chi();
     //return 100 * 0.5 * ( 1 + std::exp( -crystal.chi() ) );
     rewardCalls ++;
     return 1.0 / (crystal.relativeAvgInfid() + 1e-8);
    }
    else
    {
      return 0;
    }
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
  
  int getRewardCalls() const
  {
    return rewardCalls;
  }

};

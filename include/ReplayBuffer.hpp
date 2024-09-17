#pragma once

#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
#include <random>

//Includes the classes for the states, steps, and buffer of the Q learning protocol.

template <class State> 
using Step = std::tuple<State, int, State, double, bool>;

template <class State>
using Sample = std::tuple<std::vector<State>, std::vector<int>, 
  std::vector<State>, std::vector<double>, std::vector<bool>>;

template <class State>
class ReplayBuffer {

  size_t memorySize = 100000;
  size_t batchSize = 64;

  int stepPtr = 0;
  std::vector<Step<State>> buffer;

public:

  size_t size() const { return buffer.size(); }
//Push makes the 'Step' tuple as described above. 
  void push(const State& state, const int action, const State& next_state, 
    const double reward, const bool done)
  {
    Step<State> step = std::make_tuple(state, action, next_state, reward, done);
    if (buffer.size() < memorySize )
      buffer.push_back(step); //if buffer is less than memory size then add step to end of the buffer.
    else
      buffer[stepPtr] = step; //If buffer size larger than memory replace step at point stepptr.
    stepPtr = (stepPtr + 1) % memorySize;
  }
//Makes the Sample with all attributes much like push does. 
  Sample<State> sample(std::minstd_rand* g) const
  {
    std::vector<State> states;
    std::vector<int> actions;
    std::vector<State> next_states;
    std::vector<double> rewards;
    std::vector<bool> dones;
 // sampleSize is the lesser of the buffer size or bactch size.  
    size_t sampleSize = std::min(buffer.size(), batchSize);
    int samplePtr[sampleSize];

//creates array of numbers 0 to buffer.size() and shuffles them randomly.
//Then creates a pointer array of indices of arr[i].
//
    if (buffer.size() < 2 * batchSize) {
      int arr[buffer.size()];
      for (size_t i = 0; i < buffer.size(); i++)
        arr[i] = i;
      std::shuffle(arr, arr + buffer.size(), *g);
      for (size_t i = 0; i < sampleSize; i++)
        samplePtr[i] = arr[i];
    } else {
      for (size_t i = 0; i < sampleSize; i++) {
        int r;
        do {
          r = (*g)() % buffer.size();
        } while (std::find(samplePtr, samplePtr + i, r) != samplePtr + i);
        samplePtr[i] = r;
      }
    }
    for (size_t i = 0; i < sampleSize; i++) {
      states.push_back(std::get<0>(buffer[samplePtr[i]]));
      actions.push_back(std::get<1>(buffer[samplePtr[i]]));
      next_states.push_back(std::get<2>(buffer[samplePtr[i]]));
      rewards.push_back(std::get<3>(buffer[samplePtr[i]]));
      dones.push_back(std::get<4>(buffer[samplePtr[i]]));
    }

    return std::make_tuple(states, actions, next_states, rewards, dones);
  }

};

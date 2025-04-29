#pragma once

#include <torch/torch.h>
#include <map>
#include <random>
#include <sstream>
#include <iomanip>

#include <chrono>
#include <iostream>

#include "Network.hpp"
#include "ReplayBuffer.hpp"
#include "Translate.hpp"

void validate_model_io(const torch::nn::Module& model, int expected_input_size, int expected_output_size) {
    int input_size = -1;
    int output_size = -1;

    std::vector<std::pair<std::string, torch::Tensor>> linear_weights;

    for (const auto& param : model.named_parameters()) {
        const std::string& name = param.key();

        // Look for weight tensors of Linear layers (usually ends with "weight")
        if (name.find("weight") != std::string::npos) {
            const auto& size = param.value().sizes();

            // Linear layers have 2D weight tensors: [out_features, in_features]
            if (size.size() == 2) {
                linear_weights.emplace_back(name, param.value());
            }
        }
    }

    if (linear_weights.size() < 1) {
        throw std::runtime_error("No linear layers found in model.");
    }

    // First linear layer's input size
    input_size = linear_weights.front().second.size(1); // [out, **in**]
    output_size = linear_weights.back().second.size(0); // [**out**, in]

    if (input_size != expected_input_size) {
        throw std::runtime_error("Input size mismatch: expected " +
            std::to_string(expected_input_size) + ", got " + std::to_string(input_size));
    }

    if (output_size != expected_output_size) {
        throw std::runtime_error("Output size mismatch: expected " +
            std::to_string(expected_output_size) + ", got " + std::to_string(output_size));
    }

    std::cout << "Model input/output size validated: " 
              << input_size << " -> " << output_size << std::endl;
}


template <class State, int numAction>
class Agent {

  static constexpr double learningRate = 0.001;
  static constexpr double gamma = 0.95;
  static constexpr int numHidden = 128;//64;
  static constexpr int minSampleSize = 20;
  static constexpr double tau = 0.9;//0.8375;//0.8;

  Network dqn;
  Network target;

  ReplayBuffer<State> replayBuffer;
  torch::optim::Adam *tdOptimizer;
  std::minstd_rand *generator;

  torch::Tensor tdLoss(const Sample<State>& sample);

public:

  Agent()
  {
    std::random_device rd;
    generator = new std::minstd_rand(rd());
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));

    dqn = Network(State().size(), numHidden, numAction);
    dqn->initializeParameters();
    tdOptimizer = new torch::optim::Adam(dqn->parameters(),
      torch::optim::AdamOptions(learningRate));

    target = Network(State().size(), numHidden, numAction);
    for (auto& p : target->named_parameters()) {
      torch::NoGradGuard no_grad;
      p.value().copy_(dqn->named_parameters()[p.key()]);
    }
  }

  ~Agent()
  {
    free(generator);
    free(tdOptimizer);
  }

  void push(const State& state, const int action, const State& next_state,
    const double reward, const bool done)
  {
    replayBuffer.push(state, action, next_state, reward, done);
  }

  int proposeAction(const State& state, const double epsilon);

  double learn();

  void save(std::string fileName) {
    torch::save(dqn, fileName + ".dqn");
    torch::save(target, fileName + ".tgt");
  }

  void load(std::string fileName) {
    torch::load(dqn, fileName + ".dqn");
    torch::load(target, fileName + ".tgt");

    int nInput = State().size();
    int nOutput = numAction;

    std::cout << "Going to validate dqn network input/output size." << std::endl;
    validate_model_io(*dqn, nInput, nOutput);
    std::cout << "Going to validate target network input/output size." << std::endl;
    validate_model_io(*target, nInput, nOutput);
    exit(0);
  }

  double getTimeInPush()
  {
	  return replayBuffer.getTimeInPush();
  }
  double getTimeInSample()
  {
	  return replayBuffer.getTimeInSample();
  }
};

int optimumPolicy(const std::vector<double>& qValue)
{
  std::vector<double>::const_iterator choice =
    std::max_element(qValue.begin(), qValue.end());
  return std::distance(qValue.begin(), choice);
}

int randomPolicy(size_t numAction, std::minstd_rand* generator)
{
  return (*generator)() % numAction;
}

template <class State, int numAction>
int Agent<State, numAction>::proposeAction(const State& state, 
    const double epsilon)
{
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  if (distribution(*generator) > epsilon) {
    std::vector<double> qValue = toVector(
      dqn->forward(toTensor(state).to(torch::kDouble)));
    return optimumPolicy(qValue);
  }
  else {
    return randomPolicy(numAction, generator);
  }
}

template <class State, int numAction>
torch::Tensor Agent<State, numAction>::tdLoss(const Sample<State>& sample)
{
  torch::Tensor states, actions, next_states, rewards, dones;
  torch::Tensor qVals, next_actions, next_qVals, Y;

  std::tie(states, actions, next_states, rewards, dones) = toTensor(sample);
  qVals = dqn->forward(states).gather(1, actions);
  next_actions = dqn->forward(next_states).argmax(1, true);
  next_qVals = target->forward(next_states).gather(1, next_actions);
  Y = rewards + gamma * dones.logical_not() * next_qVals;

  return torch::mse_loss(qVals, Y.detach());
}

template <class State, int numAction>
double Agent<State, numAction>::learn()
{
  if (replayBuffer.size() < minSampleSize)
    return 0;
  Sample<State> sample = replayBuffer.sample(generator);
  torch::Tensor loss = tdLoss(sample);
  tdOptimizer->zero_grad();
  loss.backward(); 
  tdOptimizer->step();

  for (auto &p : target->named_parameters()) {
    torch::NoGradGuard no_grad;
    p.value().copy_(tau * p.value() + (1 - tau)
      * dqn->named_parameters()[p.key()]);
  }

  return loss.item<double>();
}

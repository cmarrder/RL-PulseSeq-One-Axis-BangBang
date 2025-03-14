#include <iostream>
#include <fstream>

#include "Agent.hpp"
#include "Environment.hpp"
#include "Greedy.hpp"
#include "Param.hpp"

constexpr int numTrial = 100;//250;//400;
constexpr int numEpisode = 100;

int main()
{ 

  std::string paramFile = "/home/charlie/Documents/ml/CollectiveAction/param.txt";
  Param param;
  param = getParam(paramFile);
  
  Environment environment(param);
  //Environment environment;
  Agent<Feature, numAction> agent;
  
  //agent.load("/home/charlie/Documents/ml/CollectiveAction/data/model");
  
  std::vector<int> actionRecord;
  std::vector<double> rewardHistory;
  std::vector<double> lossHistory;
  double maxReward = 0;
  double maxFid = 0;

  for (int trial = 0; trial < numTrial; trial++) {
    double epsilon = epsilonByProportion(trial / (double)numTrial);
    double avgLoss = 0;
    double maxRewardTrial = 0;
    double maxFidTrial = 0;
    int lossCtr = 0;
    for (int episode = 0; episode < numEpisode; episode++) {
      environment.reset();
      Feature state = environment.state();
      bool done = environment.done();
      while (!done) {
        int action = agent.proposeAction(state, epsilon);
        environment.applyAction(action);
        double reward = environment.reward();
        Feature next_state = environment.state();
        done = environment.done();
        agent.push(state, action, next_state, reward, done);
        avgLoss += agent.learn();
        lossCtr++;
        state = next_state;
      }
      maxRewardTrial = std::max(maxRewardTrial, environment.reward());
      maxFidTrial = maxRewardTrial * (1 + 1e-8) / (1 + maxRewardTrial); // Don't need this but helpful
      if (maxReward < environment.reward()) {
        maxReward = environment.reward();
        maxFid = maxReward * (1 + 1e-8) / (1 + maxReward); // Don't need this but helpful
        actionRecord = environment.actionRecord();
      std::cout << "Chi: " << environment.getChi() << std::endl;
      std::cout << "Sum: " << environment.getSum() << std::endl;
      }
    }

    rewardHistory.push_back(maxRewardTrial);
    lossHistory.push_back(avgLoss / lossCtr);

    if (param.verbose) {
      const auto default_precision {std::cout.precision()};
      std::cout << std::setprecision(3);
      std::cout << std::setw(3) << trial 
        << " Epsilon: " << std::setw(7) << epsilon 
        << " Avgloss: " << std::setw(8) << avgLoss / lossCtr 
        << " MaxRewardTrial: " << std::setw(8) << maxRewardTrial
        << " MaxReward: " << std::setw(8) << maxReward 
        << " MaxFidTrial: " << std::setw(8) << maxFidTrial
        << " MaxFid: " << std::setw(8) << maxFid 
        << std::endl;
      std::cout << std::setprecision(default_precision);
    }

  }
  //agent.save("/home/charlie/Documents/ml/CollectiveAction/data/model");

  // Print number of times we need to calculate average infidelity.
  std::cout << "Number of times we needed to calculate average fidelity: " << environment.getRewardCalls() << std::endl;
  
  // Apply optimal action sequence and write it to file. Print if verbose.
  std::ofstream out_action(param.oDir + "/action.txt");
  std::ofstream out_fid(param.oDir + "/fid.txt");
  std::ofstream out_chi(param.oDir + "/chi.txt");

  environment.reset();
  if (param.verbose) {
    std::cout << std::endl << "Initial state:" << std::endl << std::endl; 
    std::cout << environment.state().transpose() << std::endl;
  }
  // Write initial fidelity and chi
  out_fid << environment.getFid() << std::endl;
  out_chi << environment.getChi() << std::endl;
  for (auto &p : actionRecord) {
    environment.applyAction(p);
    if (param.verbose) {
      std::cout << "Action: " << p << std::endl;
      std::cout << "Next state:" << std::endl; 
      std::cout << environment.state().transpose() << std::endl << std::endl;
      std::cout << std::endl << "Fid: " << environment.getFid() << std::endl;
      std::cout << std::endl << "Chi: " << environment.getChi() << std::endl;
    }
    // Write action chosen, new fidelity, and chi
    out_action << p << std::endl;
    out_fid << environment.getFid() << std::endl;
    out_chi << environment.getChi() << std::endl;
  }
  if (param.verbose) {
    std::cout << std::endl << "Reward: " << environment.reward() << std::endl;
  }
  out_action.close();
  out_fid.close();
  out_chi.close();

  // Write best final state to file.
  std::ofstream out_state(param.oDir + "/state.txt");
  out_state << environment.state() << std::endl;
  out_state.close();

  // Write reward and loss histories to file.
  std::ofstream out_reward(param.oDir + "/reward.txt");
  for (auto &p : rewardHistory) {
     out_reward << p << std::endl;
  }
  out_reward.close();

  std::ofstream out_loss(param.oDir + "/loss.txt");
  for (auto &p : lossHistory) {
     out_loss << p << std::endl;
  }
  out_loss.close();

  std::ofstream out_filter(param.oDir + "/bestFilter.txt");
  out_filter << environment.getFilter() << std::endl;
  out_filter.close();

  std::cout << "final chi" << std::endl << environment.getChi() << std::endl; 

  exit(0);
}

#pragma once

#include <torch/torch.h>

struct NetworkImpl : torch ::nn::Module
{
  NetworkImpl() = default;
  NetworkImpl(const int numInput, const int numHidden, const int numAction) :
    l1(register_module("l1", torch::nn::Linear(numInput, numHidden))),
    l2(register_module("l2", torch::nn::Linear(numHidden, numHidden))),
    l3(register_module("l3", torch::nn::Linear(numHidden, numAction)))
  {
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = l1->forward(x);
    x = torch::relu(x);
    x = l2->forward(x);
    x = torch::relu(x);
    x = l3->forward(x);
    x = torch::relu(x);
    return x;
  }

  torch::nn::Linear l1{nullptr};
  torch::nn::Linear l2{nullptr};
  torch::nn::Linear l3{nullptr};

  void initializeParameters()
  {
    torch::nn::init::xavier_uniform_(named_parameters() ["l1.weight"], 0.5);
    torch::nn::init::xavier_uniform_(named_parameters() ["l2.weight"], 0.5);
    torch::nn::init::xavier_uniform_(named_parameters() ["l3.weight"], 0.5);
    torch::nn::init::uniform_(named_parameters() ["l1.bias"]);
    torch::nn::init::uniform_(named_parameters() ["l2.bias"]);
    torch::nn::init::uniform_(named_parameters() ["l3.bias"]);
    //torch::set_num_threads(16);
  }

};

TORCH_MODULE(Network);

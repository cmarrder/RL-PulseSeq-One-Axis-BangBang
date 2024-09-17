#pragma once

#include <torch/torch.h>
#include <Eigen/Dense>

template <int N>
torch::Tensor toTensor(const Eigen::Matrix<double, N, 1>& v)
{
  return torch::from_blob(const_cast<Eigen::Matrix<double, N, 1>&>(v).data(), 
    {N}).clone();
}

template <class Feature>
torch::Tensor toTensor(const std::vector<Feature>& v)
{
  torch::Tensor result = torch::empty({(long)v.size(), Feature().size()});
  for (size_t i = 0; i < v.size(); i++)
    result.slice(0, i, i+1) = toTensor(v[i]);
  return result;
}

torch::Tensor toTensor(const std::vector<int>& v)
{
  return torch::from_blob(const_cast<std::vector<int>&>(v).data(),
    {(long)v.size(), 1}, torch::kInt).clone().to(torch::kInt64);
}

torch::Tensor toTensor(const std::vector<double>& v)
{
  return torch::from_blob(const_cast<std::vector<double>&>(v).data(),
    {(long)v.size(), 1}).clone();
}

torch::Tensor toTensor(const std::vector<bool>& v)
{
  bool a[v.size()];
  std::copy(v.begin(), v.end(), a);
  return torch::from_blob(a, {(long)v.size(), 1}, torch::kBool).clone();
}

template <class State>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
    torch::Tensor> toTensor(const Sample<State>& sample)
{
  return std::make_tuple(
    toTensor(std::get<0>(sample)), 
    toTensor(std::get<1>(sample)), 
    toTensor(std::get<2>(sample)), 
    toTensor(std::get<3>(sample)), 
    toTensor(std::get<4>(sample)));
}

std::vector<double> toVector(const torch::Tensor& t)
{
  torch::Tensor tmp = t.contiguous();
  return std::vector<double>(tmp.data_ptr<double>(), tmp.data_ptr<double>()
    + tmp.numel());
}

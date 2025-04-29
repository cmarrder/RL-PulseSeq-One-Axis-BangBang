#pragma once

#include <string>
#include <iostream>
#include <fstream>

typedef struct Param {
  std::string oDir;
  bool verbose;
  double noiseParam1;
  double noiseParam2;
  double eta1;
  double maxJ;
} Param;

std::ostream& operator<< (std::ostream& o, const Param& param)
{
  o << "oDir: " << param.oDir << std::endl;
  o << "verbose: " << param.verbose << std::endl;
  o << "noiseParam1: " << param.noiseParam1 << std::endl;
  o << "noiseParam2: " << param.noiseParam2 << std::endl;
  o << "eta1: " << param.eta1 << std::endl;
  o << "maxJ: " << param.maxJ << std::endl;
  return o;
}

//Param getParam()
Param getParam(std::string& pfile)
{
  // pfile is the file containing parameter values
  Param param;

  //ifstream configInput("../param.txt");
  std::ifstream configInput(pfile);
  std::string dummy;

  while (!configInput.eof()) {
    configInput >> dummy;
    if (configInput.eof()) break;
    if (!configInput.good()) {
      std::cout << "Bad read in input file" << std::endl;
      exit(-1);
    }
    if (dummy.compare("oDir") == 0) {
      configInput >> param.oDir;
    }
    else if (dummy.compare("verbose") == 0) {
      configInput >> param.verbose;
    }
    else if (dummy.compare("noiseParam1") == 0) {
      configInput >> param.noiseParam1;
    }
    else if (dummy.compare("noiseParam2") == 0) {
      configInput >> param.noiseParam2;
    }
    else if (dummy.compare("eta1") == 0) {
      configInput >> param.eta1;
    }
    else if (dummy.compare("maxJ") == 0) {
      configInput >> param.maxJ;
    }
    else {
      //cout << "Error: invalid label " << dummy << " in param.txt" << endl;
      std::cout << "Error: invalid label " << dummy << " in " << pfile << std::endl;
      exit(-1);
    }
  }
  configInput.close();

  return param;
}

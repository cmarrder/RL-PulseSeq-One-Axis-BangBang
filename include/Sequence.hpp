#pragma once

#include <complex>
#include <Eigen/Dense>

using namespace Eigen;

const int nPulse = 4;
const double maxTime = 1;
const int nTimeStep = 1000; // Number of time steps in mesh.
const int nZero = 2000;
//typedef Matrix<Pulse, nPulse, 1> Sequence;

class Pulse {

  double centerTime;

  public:
  
    Pulse() {};
    //Pulse(const double t) : t(t) {}
    Pulse(double t) {
      centerTime = t;
    }

    double getCenterTime() const { return centerTime; }

    void updateCenterTime(double t) { centerTime = t; }
};

class Sequence {
  
  // Make container for the pulses:
  Matrix<Pulse, nPulse, 1> pulses;

  public:

    // Constructor
    Sequence() {};
    Sequence(VectorXd centerTimes) {
      updateCenterTimes(centerTimes);
    }

    void updateCenterTimes(VectorXd centerTimes)
    {
      for (int i = 0; i < nPulse; i++)
      {
        pulses(i).updateCenterTime(centerTimes(i));
      }
    }

    VectorXd getCenterTimes() const
    {
      VectorXd pulseCenters = VectorXd::Zero(nPulse);
      for (int i = 0; i < nPulse; i++)
      {
        pulseCenters(i) = pulses(i).getCenterTime();
      }
     return pulseCenters; 
    }
};


VectorXd PDDCenterTimes()
{
  // Initialize pulse center timings to be uniformly spaced from time 0 to maxTime.
  VectorXd pulseCenter = VectorXd::LinSpaced(nPulse + 2, 0, maxTime).segment(1, nPulse);
  return pulseCenter;
}


VectorXd CPMGCenterTimes()
{
  // Initialize pulse center timings to be those of the CPMG sequence.
  VectorXd idxs = VectorXd::LinSpaced(nPulse, 1.0, nPulse);
  double cpmgTau = maxTime / 2.0 / nPulse;
  VectorXd pulseCenter = ( ( ( 2.0 * idxs.array() - 1) * cpmgTau ) ).matrix();
  return pulseCenter;
}

VectorXd UDDCenterTimes()
{
  // Initialize pulse center timings to be those of the UDD sequence.
  VectorXd idxs = VectorXd::LinSpaced(nPulse, 1.0, nPulse);
  VectorXd pulseCenter = ( ( ( idxs.array() * M_PI / (2 * nPulse + 2) ).sin() ).square() ).matrix();
  return pulseCenter;
}
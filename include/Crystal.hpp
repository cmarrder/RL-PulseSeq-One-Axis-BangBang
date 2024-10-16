#pragma once

#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/FFT>
#include "Action.hpp"
#include "Sequence.hpp"
#include "Param.hpp"

constexpr std::complex<double> I(0.0, 1.0);
constexpr double cutoffEps = 1e-12;

using namespace Eigen;

typedef Matrix<double, nPulse, 1> Feature;

VectorXd heaviside(const VectorXd& x)
{
  return (x.array() < 0).select(VectorXd::Zero(x.size()), 
    VectorXd::Ones(x.size()));
}

VectorXd reciprocal(const VectorXd& x)
{
  VectorXd zeroRemoved = (x.array().abs() < 1e-12).select(VectorXd::Constant(x.size(), 1e12), x);
  return 1 / zeroRemoved.array();
}

class Crystal {

  Sequence sequence;
  VectorXd freq;
  VectorXd sOmega;
  double dFreq;
  double dTime;
  int idxMaxFreq;
  int nFreq;
  int cutoffIdx;
  int nTimePts;
  double noiseParam1;
  double noiseParam2;

  VectorXd initialCenterTimes()
  {
    // Wrapper function used to choose the initial pulse center times.
    return PDDCenterTimes();
  }
  
  VectorXd computeSOmega() const
  {
    VectorXd somega = VectorXd::Zero(nFreq);
   
     
    //// FERMI-DIRAC:
    VectorXd ONE = VectorXd::Ones(nFreq);
    VectorXd v = (freq - noiseParam1 * ONE) / noiseParam2;
    somega = ONE.array() / (v.array().exp() + ONE.array());
    

    /*
    //// SUM OF LORENTZIANS:
    VectorXd centers = peakLocCPMG(); // Place Lorentzian at each peak of CPMG Filter function
    int nPeaks = centers.size(); 
    VectorXd fwhms = VectorXd::Ones(nPeaks).array() / 2;
    VectorXd heights = pow(centers.array(), -2);
    //// Divide output by number of elements in centers to normalize.
    somega = lorentzians(freq, centers, fwhms, heights);
    */
   
    /* 
    //// STEP FUNCTION:
    double cpmgPeakFreq = nPulse / (2 * maxTime); // The first peak frequency of the CPMG filter function
    double bandMin = noiseParam1 * cpmgPeakFreq;
    double bandMax = noiseParam2 * cpmgPeakFreq;
    if (bandMax > freq(nFreq - 1))
    {
      std::cout << "Error: Noise cutoff larger than max frequency!";
    }
    somega = reciprocal(freq).array() * (heaviside(freq.array() - 0.0) - heaviside(freq.array() - bandMin) + heaviside(freq.array() - bandMax) - heaviside(freq.array() - freq(nFreq - 1))).array();
    */

    return somega; 
  }

  VectorXcd computeFOmega() const
  {
    VectorXd ctrlSig = computeCtrlSig();
    VectorXd ctrlSigPadded = VectorXd::Zero(nTimePts);
    ctrlSigPadded.head(nTimeStep) = ctrlSig;
    
    VectorXcd fourier_ft = VectorXcd::Zero(nTimePts);
    FFT<double> fft;
    fft.fwd(fourier_ft, ctrlSigPadded);
    //return fourier_ft(seq(0, idxMaxFreq)); // Only consider the positive freqs
    return dTime * fourier_ft(seq(0, idxMaxFreq)); // Only consider the positive freqs. Normalize by multiplying by dTime.

  }

  VectorXd computeFOmegaAbs2() const
  {
    VectorXcd fourier_ft = computeFOmega();
    //return (fourier_ft(seq(0, idxMaxFreq))).cwiseAbs2(); // Only consider the positive freqs
    return fourier_ft.cwiseAbs2();
  }


  VectorXd computeCtrlSig() const
  {
    // Compute control signal f(t) using the pulse sequence.
    VectorXd controlSignal = VectorXd::Ones(nTimeStep);
    double tau = maxTime / (nTimeStep);
    int pulseCounter = 0;
    VectorXd pulseCenters = sequence.getCenterTimes();
    double currentCenter = pulseCenters(pulseCounter);
    double sign = 1.0;

    for (int k = 0; k < nTimeStep; k++)
    {
      while (currentCenter > k * tau && currentCenter <= (k+1) * tau)
      {
	// Increment the count of pulses in the current bin.
	sign = -sign;
	// Increment the index of the pulse to consider.
        pulseCounter = pulseCounter + 1;
	// If the max number of pulses has been reached, end the while loop.
	if (pulseCounter == nPulse) {
	  break;
	}
        // Use the next pulse center.
        currentCenter = pulseCenters(pulseCounter);

      }
      controlSignal(k) = sign; 
    }
    return controlSignal;
  }

  int getCutoffIndex()
  {
    for (int k = nFreq - 1; k > -1; k--)
    {
      if (sOmega(k) / std::pow(freq(k), 2) > cutoffEps)
      {
	std::cout << "cutOff Index: "<< k << std::endl;
	std::cout << "max Index: "<< nFreq - 1 << std::endl;
        return k;
      }
    }
    return nFreq - 1;
  }

  void writeInitial(std::string& dir)
  {
    std::ofstream out_freq(dir + "/freq.txt");
    out_freq << freq << std::endl;
    out_freq.close();

    std::ofstream out_sOmega(dir + "/sOmega.txt");
    out_sOmega << sOmega << std::endl;
    out_sOmega.close();

    std::ofstream out_noiseParam1(dir + "/noiseParam1.txt");
    out_noiseParam1 << noiseParam1 << std::endl;
    out_noiseParam1.close();

    std::ofstream out_noiseParam2(dir + "/noiseParam2.txt");
    out_noiseParam2 << noiseParam2 << std::endl;
    out_noiseParam2.close(); 

    std::ofstream out_maxTime(dir + "/maxTime.txt");
    out_maxTime << maxTime << std::endl;
    out_maxTime.close();

    std::ofstream out_nTimeStep(dir + "/nTimeStep.txt");
    out_nTimeStep << nTimeStep << std::endl;
    out_nTimeStep.close();

    std::ofstream out_nPulse(dir + "/nPulse.txt");
    out_nPulse << nPulse << std::endl;
    out_nPulse.close();

  }

  public:

  Crystal() {};

  Crystal(Param& param)
  {
    noiseParam1 = param.noiseParam1;
    noiseParam2 = param.noiseParam2;
    int nZero = 32768 - (nTimeStep + 1);//8 * (nTimeStep + 1); // Number of zeros for padding.
    nTimePts = (nTimeStep + 1) + nZero; // Make a power of 2 to make FFT faster
    dTime = maxTime / nTimeStep;

    idxMaxFreq = (nTimePts - 1 - (nTimePts - 1) % 2) / 2;

    VectorXd range = VectorXd::LinSpaced(idxMaxFreq + 1, 0, idxMaxFreq); // range of integers from 0 to idxMaxFreq
    
    freq = range / (nTimePts * dTime);
    dFreq = freq(1) - freq(0);
    nFreq = freq.size();
    
    sOmega = computeSOmega();

    cutoffIdx = getCutoffIndex();
    
    Sequence sequence(initialCenterTimes());

    writeInitial(param.oDir);
  }

  void reset()
  {
    sequence.updateCenterTimes(initialCenterTimes());
  }

  void step(int action)
  {
    VectorXd oldTimes = sequence.getCenterTimes();
    sequence.updateCenterTimes(actionList[action](oldTimes));
  }

  Feature feature() const
  {
    Feature feature;
    // Extract pulse center times from sequence. 
    feature = sequence.getCenterTimes();
    return feature;
  }
    
  double chi() const
  {
    VectorXd fOmegaAbs2 = computeFOmegaAbs2();
    double overlap =  (fOmegaAbs2.head(cutoffIdx + 1).transpose() * sOmega.head(cutoffIdx + 1))(0) * dFreq; // Select 0 element because technically result is vector of length 1
    return overlap;
  }

};

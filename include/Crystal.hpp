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
#include "GaussQuad.hpp"

constexpr bool useFFT = false;
constexpr double cutoffEps = 1e-6;

constexpr std::complex<double> I(0.0, 1.0);

using namespace Eigen;

typedef Matrix<double, nPulse, 1> Feature;

VectorXd heaviside(const VectorXd& x)
{
  // Heaviside step function as a function of x.
  return (x.array() < 0).select(VectorXd::Zero(x.size()), VectorXd::Ones(x.size()));
}
  
VectorXd boxcar(const VectorXd& x, const double center, const double width)
{
  // Boxcar function as a function of x.
  double left = center - width / 2.0;
  double right = center + width / 2.0;
  return heaviside(x.array() - left) - heaviside(x.array() - right);
}

VectorXd sinc(const VectorXd& x)
{
  double EPS = 1e-20;
  int dim = x.size();
  VectorXd res(dim);
  for (int i = 0; i < dim; i++)
    res(i) = (fabs(x(i)) < EPS ? 1 : sin(M_PI * x(i)) / (M_PI * x(i)));
  return res;
}

VectorXd lorentzian(const VectorXd& x, const double center, const double fwhm)
{
  // Normalized Lorentzian as a function of x.
  int nA = x.size();
  VectorXd numerator = VectorXd::Ones(nA) * fwhm / 2 / M_PI;
  VectorXd denominator = (x.array() - center).pow(2) + std::pow(fwhm / 2, 2);
  return numerator.cwiseQuotient(denominator);
}

VectorXd gaussian(const VectorXd& x, const double mu, const double sigma)
{
  // Normalized Gaussian as a function of x.
  VectorXd numerator = exp(-square(x.array() - mu) / 2 / sigma);
  double denominator = std::sqrt(2 * M_PI * sigma * sigma); 
  return numerator / denominator;
}

VectorXd lorentzians(const VectorXd& x, const VectorXd& centers, const VectorXd& fwhms, const VectorXd& heights)
{
  // Sum of Lorentzians. Each of which is normalized.
  // NOTE: centers and fwhms must have same length
  int nCenter = centers.size();
  int nA = x.size();
  VectorXd sum = VectorXd::Zero(nA);
  for (int k = 0; k < nCenter; k++)
  {
    sum += heights(k) * M_PI * (fwhms(k) / 2) * lorentzian(x, centers(k), fwhms(k));
  }
  return sum;
}

VectorXd gaussians(const VectorXd& x, const VectorXd& mus, const VectorXd& sigmas, const VectorXd& heights)
{
  // Sum of Gaussians. Each of which is normalized.
  // NOTE: centers and fwhms must have same length
  // HOW NORMALIZE???
  int nCenter = mus.size();
  int nA = x.size();
  VectorXd sum = VectorXd::Zero(nA);
  for (int k = 0; k < nCenter; k++)
  {
    sum += heights(k) * gaussian(x, mus(k), sigmas(k));
  }
  return sum;
}

VectorXd reciprocal(const VectorXd& x)
{
  // Calculate reciprocal of x. If |x| < 1e-12, then return 1e+12.
  VectorXd zeroRemoved = (x.array().abs() < 1e-12).select(VectorXd::Constant(x.size(), 1e-12), x);
  return 1 / zeroRemoved.array();
}


class Crystal {

  Sequence sequence;
  VectorXd freq;
  VectorXd weights;
  VectorXd sOmega;
  VectorXd weightedSOmega;
  VectorXd actionHarmonics;
  VectorXd actionEtas;
  double dFreq;
  double dTime;
  int idxMaxFreq;
  int nFreq;
  int cutoffIdx;
  int nTimePts;
  double noiseParam1;
  double noiseParam2;
  double eta1;
  double initialChi;
  double initialAvgInfid;

  VectorXd initialCenterTimes()
  {
    // Wrapper function used to choose the initial pulse center times.
    //return CPMGCenterTimes();
    return PDDCenterTimes();
    //VectorXd centerTimes(5);
    //centerTimes << 0, 0.25, 0.25, 0.25, 0.75;
    //return centerTimes;
  }
  
  VectorXd peakLocCPMG() const
  {
    // Calculate the peaks of the CPMG filter function 
    int nPeaks = nPulse; // Currently only use nPulse many peaks. Can add more if need be.
    VectorXd range = VectorXd::LinSpaced(nPeaks, 0, nPeaks - 1); // range of integers from 0 to nPeaks - 1
    double denom = maxTime / nPulse;
    VectorXd centers = (range.array() + 0.5) / denom;
    return centers;
  }
  
  VectorXd computeSOmega(VectorXd frequencies) const
  {
    VectorXd somega = VectorXd::Zero(nFreq);
    
    //// UNCOMMENT THE NOISE MODEL YOU WANT BELOW: 
    
    /*
    //// FERMI-DIRAC:
    VectorXd ONE = VectorXd::Ones(frequencies.size());
    VectorXd v = (frequncies - noiseParam1 * ONE) / noiseParam2;
    somega = ONE.array() / (v.array().exp() + ONE.array());
    */

             
    //// 1/f NOISE:
    somega = reciprocal(frequencies);
    

    /*     
    //// LORENTZIAN:
    double flipRate = (1.0 / 10.0) * (1.0 / maxTime);
    double fwhm = 4 * flipRate;
    //// Divide output by number of elements in centers to normalize.
    somega = lorentzian(freq, 0, fwhm);
    */

    /* 
    //// SUM OF LORENTZIANS:
    //VectorXd centers = peakLocCPMG(); // Place Lorentzian at each peak of CPMG Filter function
    //VectorXd fwhms = VectorXd::Ones(nPeaks).array() / 2;
    //VectorXd heights = pow(centers.array(), -2);
    double cpmgPeakFreq = nPulse / (2 * maxTime); // The first peak frequency of the CPMG filter function
    //Vector2d centers(0.0, 0.7 * cpmgPeakFreq); // Place Lorentzian at each peak of CPMG Filter function
    Vector2d centers(0.0, 0.7 * cpmgPeakFreq); // Place Lorentzian at origin and some fraction of location of peak of CPMG Filter function
    int nPeaks = centers.size(); 
    Vector2d fwhms(1.0 / 2.0, 1.0 / 4.0); // MAKE SURE FWHM IS NOT SMALLER THAN KEY FREQUENCIES IN TIME MESH
    VectorXd heights = reciprocal(2 * M_PI * fwhms); 
    //// Divide output by number of elements in centers to normalize.
    somega = lorentzians(frequencies, centers, fwhms, heights) / nPeaks;
    */

    /*
    //// SUM OF GAUSSIANS:
    VectorXd centers = peakLocCPMG(); // Place Gaussian at each peak of CPMG Filter function
    centers = (centers.array() + centers(0) / 2).matrix(); // Shift so that noise peaks are in between CPMG filter function peaks
    int nPeaks = centers.size(); 
    VectorXd sigmas = VectorXd::Ones(nPeaks).array() / 2;
    VectorXd heights = pow(centers.array(), -2);
    somega = gaussians(frequencies, centers, sigmas, heights);
    */
   
    /* 
    //// BOXCAR FUNCTION:
    double cpmgPeakFreq = nPulse / (2 * maxTime); // The first peak frequency of the CPMG filter function
    double freqResolution = 1.0 / maxTime;
    double bandWidth = noiseParam1 * freqResolution;
    double bandCenter = noiseParam2 * cpmgPeakFreq;
    if (noiseParam2 < 1.0)
    {
      std::cout << "Error: Noise bandwidth is less than the frequency resolution!" << std::endl;
    }
    //somega = reciprocal(frequencies).array() * (1.0 - boxcar(frequencies, bandCenter, bandWidth)).array();
    //somega = 1.0 - boxcar(frequencies, bandCenter, bandWidth).array();
    // Noise which is a sum of two boxcars, each with distinct centers and widths.
    double boxcar1Right = bandCenter - bandWidth / 2; // Right boundary of 1st boxcar
    double center1 = (boxcar1Right + 0) / 2;
    double width1 = boxcar1Right - 0;
    double boxcar2Left = bandCenter + bandWidth / 2; // Left boundary of 2nd boxcar
    double center2 = (frequencies(idxMaxFreq) + boxcar2Left) / 2;
    double width2 = frequencies(idxMaxFreq) - boxcar2Left;
    somega = boxcar(frequencies, center1, width1) + boxcar(frequencies, center2, width2);
    */
    
    /* 
    //// STEP FUNCTION
    //double freqResolution = 1.0 / maxTime;
    //double bandWidth = noiseParam1 * freqResolution;
    double cpmgPeakFreq = nPulse / (2 * maxTime); // The first peak frequency of the CPMG filter function
    double bandWidth = noiseParam1 * cpmgPeakFreq;
    somega = heaviside(frequencies.array()) - heaviside(frequencies.array() - bandWidth);
    */

    return somega; 
  }

  VectorXcd computeFOmega() const
  {
    if (useFFT == true)
    {
        VectorXd ctrlSig = computeCtrlSig();
        VectorXd ctrlSigPadded = VectorXd::Zero(nTimePts);
        ctrlSigPadded.head(nTimeStep) = ctrlSig;
        
        VectorXcd fourier_ft = VectorXcd::Zero(nTimePts);
        FFT<double> fft;
        fft.fwd(fourier_ft, ctrlSigPadded);
        return dTime * fourier_ft(seq(0, idxMaxFreq)); // Only consider the positive freqs. Normalize by multiplying by dTime.
    }
    else
    {
        VectorXd timeWithEnds = VectorXd::Zero(nPulse + 2);
        timeWithEnds(0) = 0;
        timeWithEnds(nPulse + 1) = maxTime;
        timeWithEnds(seq(1, nPulse)) = sequence.getCenterTimes();
        
        VectorXd boxcarWidths = timeWithEnds(seq(1, nPulse+1)) - timeWithEnds(seq(0, nPulse));
        VectorXd boxcarCenters = (timeWithEnds(seq(1, nPulse+1)) + timeWithEnds(seq(0, nPulse))) / 2;

        
        VectorXcd fourier_f_of_t = VectorXcd::Zero(nFreq);
        for (int k = 0; k < nPulse + 1; k++) {
                VectorXd boxFT = sinc(freq * boxcarWidths(k));
                fourier_f_of_t += (std::pow(-1, k) * (-I * 2.0 * M_PI * freq * boxcarCenters(k)).array().exp() * boxcarWidths(k) * boxFT.array()).matrix();
        }
        return fourier_f_of_t;
    }
  }

  VectorXd computeFOmegaAbs2() const
  {
    VectorXcd fourier_ft = computeFOmega();
    return fourier_ft.cwiseAbs2();
  }

  VectorXd computeCtrlSig() const
  {
    /*
    Schematic to illustrate the binning with nTime=3 for simplicity:
    time points: t0          t1          t2          t3
    bins:        [    b0      )
                              [    b1     )
			                  [    b2     ]
    Here, bracket [ signifies inclusive bound, parentheses ( signifies exclusive bound.
    */

    // Compute control signal f(t) using the pulse sequence.
    VectorXd controlSignal = VectorXd::Ones(nTimeStep);
    VectorXd pulseCenters = sequence.getCenterTimes();
    double binWidth = maxTime / (nTimeStep);


    ///////////////////////////////////
    /// NEW VERSION:

    //VectorXi binIdxs = (floor(pulseCenters.array() / binWidth)).cast<int>();
    //// If an element of binIdxs is equal to nTimeStep, make it nTimeStep-1.
    //binIdxs = (binIdxs.array() == nTimeStep).select(VectorXi::Constant(nPulse, nTimeStep-1), binIdxs);
    //
    //int maxBinIdx = nTimeStep - 1;
    //for (int k = 0; k < nPulse; k++)
    //{
    //  int b = binIdxs(k);
    //  controlSignal(seq(b, maxBinIdx)) = -controlSignal(seq(b, maxBinIdx));
    //}
    /////////////////////////////////

    //// OLD VERSION:
    VectorXi binIdxs = (floor(pulseCenters.array() / binWidth)).cast<int>();
    // If an element of binIdxs is equal to nTimeStep, make it nTimeStep-1.
    binIdxs = (binIdxs.array() == nTimeStep).select(VectorXi::Constant(nPulse, nTimeStep-1), binIdxs);
    VectorXi binCounts = VectorXi::Zero(nTimeStep);
    double sign = 1.0;
    
    for (int k = 0; k < nPulse; k++)
    {
      binCounts(binIdxs(k)) += 1;
    }
    for (int i = 0; i < nTimeStep; i++)
    {
      if (binCounts(i) % 2 == 1)
      {
        sign = -sign;
      }
      controlSignal(i) = sign;
    }
    
    return controlSignal;
  }

  int getCutoffIndex(VectorXd somega, VectorXd frequencies)
  {
    if (somega.size() != frequencies.size())
    {
      std::cout << "In getCutoffIndex, somega and frequencies of different sizes." << std::endl;
    }
    int numF = frequencies.size();
    for (int k = numF - 1; k > -1; k--)
    {
      // Based on upper bounds on filter, do we need factor of nPulse in numerator of following expression?
      if (somega(k) / std::pow(frequencies(k), 2) > cutoffEps)
      {
        return k;
      }
    }
    return numF - 1;
  }

  double getCutoffFrequency()
  {
    // HERE
  }

  void initializeTimeFreqData()
  {
    dTime = maxTime / nTimeStep; // Global variable
    if (useFFT == true)
    {
      nTimePts = (nTimeStep + 1) + nZero; // Global variable
      idxMaxFreq = (nTimePts - 1 - (nTimePts - 1) % 2) / 2; // Global variable. Index of max positive frequency in fft freq mesh.
      VectorXd range = VectorXd::LinSpaced(idxMaxFreq + 1, 0, idxMaxFreq); // range of integers from 0 to idxMaxFreq
      
      freq = range / (nTimePts * dTime); // Global variable
      nFreq = freq.size(); // Global variable
      dFreq = freq(1) - freq(0); // Global variable
    } 
    else
    {
      nFreq = 1000; // Global variable. Should be factor of 5.
      double nyquistFreq = 0.5 / dTime; 
      dFreq = nyquistFreq / (nFreq + 1); // Make temporary frequency increment. Will be finalized later. 
      VectorXd tempFreqs = VectorXd::LinSpaced(nFreq, 0, nyquistFreq);
      VectorXd tempSOmega = computeSOmega(tempFreqs);
      double maxFreq = tempFreqs(getCutoffIndex(tempSOmega, tempFreqs));

      // Gaussian quadrature points and weights
      MatrixXd gQuad = shiftedGaussQuad5(0, maxFreq, nFreq/5);
  
      //freq = VectorXd::LinSpaced(nFreq, 0, maxFreq); // Global variable
      freq = gQuad.col(0);
      weights = gQuad.col(1);

    }
  }


  void writeInitial(std::string& dir)
  {
    std::ofstream out_freq(dir + "/freq.txt");
    out_freq << freq << std::endl;
    out_freq.close();

    std::ofstream out_weights(dir + "/weights.txt");
    out_weights << weights << std::endl;
    out_weights.close();

    std::ofstream out_sOmega(dir + "/sOmega.txt");
    out_sOmega << sOmega << std::endl;
    out_sOmega.close();

    std::ofstream out_noiseParam1(dir + "/noiseParam1.txt");
    out_noiseParam1 << noiseParam1 << std::endl;
    out_noiseParam1.close();

    std::ofstream out_noiseParam2(dir + "/noiseParam2.txt");
    out_noiseParam2 << noiseParam2 << std::endl;
    out_noiseParam2.close(); 
    
    std::ofstream out_eta1(dir + "/eta1.txt");
    out_eta1 << eta1 << std::endl;
    out_eta1.close(); 

    std::ofstream out_harmonics(dir + "/harmonics.txt");
    out_harmonics << actionHarmonics << std::endl;
    out_harmonics.close(); 

    std::ofstream out_maxTime(dir + "/maxTime.txt");
    out_maxTime << maxTime << std::endl;
    out_maxTime.close();

    std::ofstream out_nTimeStep(dir + "/nTimeStep.txt");
    out_nTimeStep << nTimeStep << std::endl;
    out_nTimeStep.close();

    std::ofstream out_nPulse(dir + "/nPulse.txt");
    out_nPulse << nPulse << std::endl;
    out_nPulse.close();

    std::ofstream out_initialState(dir + "/initialState.txt");
    out_initialState << initialCenterTimes() << std::endl;
    out_initialState.close();

    std::ofstream out_initialFilter(dir + "/initialFilter.txt");
    out_initialFilter << computeFOmegaAbs2() << std::endl;
    out_initialFilter.close();

    std::ofstream out_initialSignal(dir + "/initialSignal.txt");
    out_initialSignal << computeCtrlSig() << std::endl;
    out_initialSignal.close();
  }

  public:

  Crystal() {};

  Crystal(Param& param)
  {
    noiseParam1 = param.noiseParam1;
    noiseParam2 = param.noiseParam2;
    //etaN = param.etaN; // The eta associated with the highest harmonic action, that whose harmonic number = nPulse.
    eta1 = param.eta1; // The eta associated with j=1.

    // Initialize harmonics for action functions. We need to do this in a roundabout way because
    // C++ doesn't allow comma initialization of Eigen::VectorXd instances when in a class but outside of a method.
    VectorXd harmonics(8);
    harmonics << 1, -1, 2, -2, 7, -7, 8, -8;
    actionHarmonics = harmonics;
    //actionEtas = (nPulse / actionHarmonics.array()).abs() * etaN; 
    actionEtas = (1 / actionHarmonics.array()).abs() * eta1; 

    std::cout << "actionHarmonics:" << std::endl << actionHarmonics.transpose() <<std::endl;
    std::cout << "actionEtas:" << std::endl << actionEtas.transpose() <<std::endl;
   
    initializeTimeFreqData(); 
    sOmega = computeSOmega(freq);
    weightedSOmega = sOmega.cwiseProduct(weights);

    if (useFFT == true)
    {
      cutoffIdx = getCutoffIndex(sOmega, freq);
    }

    sequence.updateCenterTimes(initialCenterTimes());

    writeInitial(param.oDir);
  }

  void reset()
  {
    sequence.updateCenterTimes(initialCenterTimes());
  }

  void step(int action)
  {
    VectorXd oldTimes = sequence.getCenterTimes();
    //sequence.updateCenterTimes(actionList[action](oldTimes)); // Used this when action functions encoded as fixed function objects.
    VectorXd newTimes = kappa(oldTimes, actionHarmonics[action], actionEtas[action]);
    sequence.updateCenterTimes(newTimes);
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
    double overlap;
    if (useFFT == true)
    {
      overlap =  (fOmegaAbs2.head(cutoffIdx).transpose() * sOmega.head(cutoffIdx))(0) * dFreq; // Select 0 element because technically result of vector product is vector of length 1
    }
    else
    {
      overlap =  (fOmegaAbs2.transpose() * weightedSOmega)(0); // Select 0 element because technically result of vector product is vector of length 1
    }
    return overlap;
  }

  double getInitialChi() const
  {
    return initialChi;
  }

  double avgFid() const
  {
    // Calculate average fidelity.
    return 0.5 * (1 + std::exp(-chi()));
  }

  double avgInfid() const
  {
    // Calculate average infidelity.
    return 1.0 - 0.5 * (1 + std::exp(-chi()));
  }

 double relativeAvgInfid() const
 {
   return avgInfid() / initialAvgInfid;
 } 

 VectorXd getSum() const
  {
    VectorXd fOmegaAbs2 = computeFOmegaAbs2();
    //double overlap =  (fOmegaAbs2.head(cutoffIdx + 1).transpose() * sOmega.head(cutoffIdx + 1))(0) * dFreq; // Select 0 element because technically result of vector product is vector of length 1
    
    return fOmegaAbs2.transpose() * sOmega;
  }

 VectorXd getFilter() const
 {
	 return computeFOmegaAbs2();
 }

};

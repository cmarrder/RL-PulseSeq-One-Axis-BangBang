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
#include <gsl/gsl_sf_expint.h> // Package for Si and Ci, the sin and cos integrals.

/*
Choose from the following noise options:

  fermi_dirac
  1_over_f_with_low_cutoff
  lorentzian
  lorentzians
  gaussians

 */
std::string noise = "lorentzians";//"1_over_f_with_low_cutoff";
constexpr bool useFFT = false;
constexpr double cutoffEps = 1e-6;

constexpr std::complex<double> I(0.0, 1.0);

using namespace Eigen;

typedef Matrix<double, nPulse, 1> Feature;

VectorXd myReciprocal(const VectorXd& x, const double epsilon)
{
  // Can't just name this function reciprocal because there is a function named reciprocal in Torch library.
  // Calculate reciprocal of x. If |x| < epsilon, then return epsilon.
  VectorXd zeroRemoved = (abs(x.array()) < epsilon).select(VectorXd::Constant(x.size(), epsilon), x);
  return inverse(zeroRemoved.array()).matrix();
}

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
  return heaviside((x.array() - left).matrix()) - heaviside((x.array() - right).matrix());
}

VectorXd sinc(const VectorXd& x)
{
  double EPS = 1e-20;
  VectorXd y = x * M_PI;
  VectorXd res = ( sin(y.array()) * myReciprocal(y, EPS).array() ).matrix();
  return res;
}

double sinc(const double x)
{
  double EPS = 1e-20;
  double y = 0.0;
  if (x == 0.0)
  {
    y = M_PI * EPS; 
  }
  else
  {
    y = M_PI * x;
  }
  return std::sin(y) / y;
}

VectorXd lorentzian(const VectorXd& x, const double center, const double fwhm)
{
  // Normalized Lorentzian as a function of x.
  int nA = x.size();
  VectorXd numerator = VectorXd::Ones(nA) * fwhm / 2 / M_PI;
  VectorXd denominator = ( (x.array() - center).pow(2) + std::pow(fwhm / 2, 2) ).matrix();
  return numerator.cwiseQuotient(denominator);
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

VectorXd gaussian(const VectorXd& x, const double mu, const double sigma, const double amplitude)
{
  // Normalized Gaussian as a function of x.
  VectorXd numerator = ( exp(-square((x.array() - mu) / sigma) / 2) ).matrix();
  double denominator = sigma * std::sqrt(2 * M_PI); 
  return amplitude * numerator / denominator;
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
    sum +=  gaussian(x, mus(k), sigmas(k), heights(k));
  }
  return sum;
}

double x_Si(const double x)
{
  return x * gsl_sf_Si(x);
}

double xSq_Ci(const double x)
{
  if (x == 0.0)
  {
    return 0.0;
  }
  else
  {
    return x * x * gsl_sf_Ci(x);
  }
}

// HERE: CONTINUE SWITCHING ARRAY BACK TO VECTOR

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
  VectorXcd FT_of_ft_term1;
  VectorXcd FT_of_ft_prefactor;
  VectorXd FT_of_ft_2_times_signs;
  
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

  VectorXd fermiDiracDistrib(VectorXd frequencies, double chemPot, double temperature)
  {
    VectorXd y = VectorXd::Zero(frequencies.size());
    VectorXd x = ( (frequencies.array() - chemPot) / temperature ).matrix();
    y = ( inverse((exp(x.array()) + 1)) ).matrix();
    return y;
  }

  VectorXd computeSOmega(VectorXd frequencies)
  {
    VectorXd somega = VectorXd::Zero(nFreq);
    
    if (noise == "fermi_dirac")
    {
      somega = fermiDiracDistrib(frequencies, noiseParam1, noiseParam2);
    }
             
    else if (noise == "1_over_f_with_low_cutoff")
    {
      // Set the frequencies below cutoffFreq to 1
      double S0 = 1.0;//noiseParam1;
      double cutoffFreq = 1.0 / maxTime;
      VectorXd lowFreqRemoved = ( abs(frequencies.array()) < cutoffFreq).select(VectorXd::Constant(frequencies.size(), 1.0), frequencies);
      somega = S0 * inverse(lowFreqRemoved.array()).matrix();
    }

    else if (noise == "lorentzian")
    {
      double flipRate = (1.0 / 10.0) * (1.0 / maxTime);
      double fwhm = 4 * flipRate;
      somega = lorentzian(freq, 0, fwhm);
    }

    else if (noise == "lorentzians")
    {
      //double cpmgPeakFreq = nPulse / (2 * maxTime); // The first peak frequency of the CPMG filter function
      //Vector2d centers(0.0, 0.7 * cpmgPeakFreq); // Place Lorentzian at origin and some fraction of location of peak of CPMG Filter function
      //int nPeaks = centers.size(); 
      //Vector2d fwhms(1.0 / 2.0, 1.0 / 4.0); // MAKE SURE FWHM IS NOT SMALLER THAN KEY FREQUENCIES IN TIME MESH
      //VectorXd heights = 10 * inverse(2 * M_PI * fwhms.array()).matrix(); 
      ////// Divide output by number of elements in centers to normalize.
      //somega = lorentzians(frequencies, centers, fwhms, heights) / nPeaks;

      VectorXd centers {{0.0, 2.0, 6.5, 18.5, 24.0, 30.5, 32.0, 34.5, 35.5, 39.0, 42.0}};
      VectorXd fwhms {{0.97791257, 0.69318029, 1.33978924, 1.22886943, 1.04032235,
	      0.57028931, 1.85807798, 2.40010327, 0.71804862, 1.9982775, 0.76180626}};
      VectorXd heights {{10.0, 7.07106781, 3.9223227, 2.32495277, 2.04124145, 1.81071492,
	      1.76776695, 1.70251306, 1.67836272, 1.60128154, 1.5430335}};

      somega = 10 * lorentzians(frequencies, centers, fwhms, heights);
    }

    else if (noise == "gaussians")
    {
      VectorXd centers = peakLocCPMG(); // Place Gaussian at each peak of CPMG Filter function
      centers = ( centers.array() + centers(0) / 2 ).matrix(); // Shift so that noise peaks are in between CPMG filter function peaks
      int nPeaks = centers.size(); 
      VectorXd sigmas = VectorXd::Ones(nPeaks) / 2;
      VectorXd heights = ( centers.array().pow(-2) ).matrix();
      somega = gaussians(frequencies, centers, sigmas, heights);
    }
   
    return somega; 
  }

  VectorXcd computeFOmega() const
  {
    // Compute Fourier transform of the switching function f(t).
    if (useFFT == true)
    {
        VectorXd ctrlSig = computeCtrlSig();
        VectorXd ctrlSigPadded = VectorXd::Zero(nTimePts);
        ctrlSigPadded.head(nTimeStep) = ctrlSig;
        
        VectorXcd FT_of_ft = VectorXcd::Zero(nTimePts);
        FFT<double> fft;
        fft.fwd(FT_of_ft, ctrlSigPadded.matrix());
        return dTime * FT_of_ft(seq(0, idxMaxFreq)).array(); // Only consider the positive freqs. Normalize by multiplying by dTime.
    }
    else
    {
        VectorXd timeWithEnds = VectorXd::Zero(nPulse + 2);
        timeWithEnds(0) = 0;
        timeWithEnds(nPulse + 1) = maxTime;
        timeWithEnds(seq(1, nPulse)) = sequence.getCenterTimes();
        
        VectorXd boxcarWidths = timeWithEnds(seq(1, nPulse+1)) - timeWithEnds(seq(0, nPulse));
        VectorXd boxcarCenters = (timeWithEnds(seq(1, nPulse+1)) + timeWithEnds(seq(0, nPulse))) / 2;

        VectorXcd FT_of_ft = VectorXcd::Zero(nFreq);
        for (int k = 0; k < nPulse + 1; k++) {
                VectorXd boxFT = sinc(freq * boxcarWidths(k));
                FT_of_ft = FT_of_ft +
			( std::pow(-1, k) * exp(-I * 2.0 * M_PI * freq.array() * boxcarCenters(k)) * boxcarWidths(k) * boxFT.array() ).matrix();
        }
        return FT_of_ft;
    }
  }

  VectorXd computeFOmegaAbs2() const
  {
    VectorXcd FT_of_ft = computeFOmega();
    return FT_of_ft.cwiseAbs2();
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
      exit(0);
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

  double getCutoffFrequency(double initialGuess)
  {
    // Determine UV Cutoff
    Vector2d S_div_freq_sq;
    Vector2d freqTestPts;

    double freqIncrement = initialGuess / 100.0;
    double freqMax = initialGuess + freqIncrement;

    // Initialize
    freqTestPts << initialGuess, freqMax;
    S_div_freq_sq = ( computeSOmega(freqTestPts).array() / pow(freqTestPts.array(), 2.0) ).matrix();

    while (S_div_freq_sq(1) / S_div_freq_sq(0) > 1e-9)
    {
      freqMax = freqMax + freqIncrement;
      freqTestPts(1) = freqMax;
      S_div_freq_sq = ( computeSOmega(freqTestPts).array() / pow(freqTestPts.array(), 2.0) ).matrix();
    }

    return freqMax;
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
      double UV_Cutoff_Guess = 0;
      if (noise == "fermi_dirac")
      {
        UV_Cutoff_Guess = noiseParam1;
      }
               
      else if (noise == "1_over_f_with_low_cutoff")
      {
        double cutoffFreq = 1.0 / maxTime;
        UV_Cutoff_Guess = cutoffFreq;
      }

      else if (noise == "lorentzian")
      {
        double flipRate = (1.0 / 10.0) * (1.0 / maxTime);
        double fwhm = 4 * flipRate;
        UV_Cutoff_Guess = fwhm / 2.0;
      }

      else if (noise == "lorentzians")
      {
        //double cpmgPeakFreq = nPulse / (2 * maxTime); // The first peak frequency of the CPMG filter function
        //Vector2d centers(0.0, 0.7 * cpmgPeakFreq); // Place Lorentzian at origin and some fraction of location of peak of CPMG Filter function
        //Vector2d fwhms(1.0 / 2.0, 1.0 / 4.0); // MAKE SURE FWHM IS NOT SMALLER THAN KEY FREQUENCIES IN TIME MESH
        //UV_Cutoff_Guess = centers(1) + fwhms(1) / 2.0;
        UV_Cutoff_Guess = 208.01139517 / 2.0 / M_PI;
      }

      else if (noise == "gaussians")
      {
        VectorXd centers = peakLocCPMG(); // Place Gaussian at each peak of CPMG Filter function
        centers = ( centers.array() + centers(0) / 2 ).matrix(); // Shift so that noise peaks are in between CPMG filter function peaks
        int nPeaks = centers.size(); 
        VectorXd sigmas = VectorXd::Ones(nPeaks) / 2;
        VectorXd heights = ( centers.array().pow(-2) ).matrix();
        UV_Cutoff_Guess = centers(nPeaks - 1) + sigmas(nPeaks - 1);
      }

      nFreq = 40000; // Global variable. Should be factor of 5.
      double maxFreq = 0;
      maxFreq = getCutoffFrequency(UV_Cutoff_Guess);
      std::cout << "maxFreq:" << std::endl << maxFreq << std::endl;
       
      // Check is FID filter is normalized, then shift window as long as S / freq**2 from UV below a cutoff, or as long as chi remains the same after freq doubling

      // Gaussian quadrature points and weights
      MatrixXd gQuad = shiftedGaussQuad5(0, maxFreq, nFreq/5);
  
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
    //
    //VectorXd harmonics(8);
    //double J = static_cast<double>(nPulse);
    //harmonics << 1, -1, 2, -2, J-1, -(J-1), J, -J;
    //
    VectorXd harmonics(6);
    double J = param.maxJ;//static_cast<double>(nPulse);
    harmonics << 1, -1, 2, -2, J, -J;
    actionHarmonics = harmonics;

    //actionEtas = ( (nPulse / actionHarmonics.array()).abs() * etaN ).matrix(); 
    actionEtas = ( (1 / actionHarmonics.array()).abs() * eta1 ).matrix(); 

    std::cout << "nPulse: " << nPulse << std::endl;
    std::cout << "actionHarmonics:" << std::endl << actionHarmonics.transpose() << std::endl;
    std::cout << "actionEtas:" << std::endl << actionEtas.transpose() << std::endl;
   
    initializeTimeFreqData(); 
    sOmega = computeSOmega(freq);
    weightedSOmega = sOmega.cwiseProduct(weights);

    if (useFFT == true)
    {
      cutoffIdx = getCutoffIndex(sOmega, freq);
    }

    sequence.updateCenterTimes(initialCenterTimes());

    writeInitial(param.oDir);

    std::cout << "Initial fid: " << avgFid() << std::endl;

  }

  void reset()
  {
    sequence.updateCenterTimes(initialCenterTimes());
  }

  void step(int action)
  {
    VectorXd oldTimes = sequence.getCenterTimes();
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

  double chi_1_over_f_low_cutoff(double S0, double freqCutoff) const
  {
    VectorXd timeWithEnds = VectorXd::Zero(nPulse + 2);
    timeWithEnds(0) = 0;
    timeWithEnds(nPulse + 1) = maxTime;
    timeWithEnds(seq(1, nPulse)) = sequence.getCenterTimes();
    double angfreqCutoff = 2 * M_PI * freqCutoff;

    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double sum4 = 0.0;

    for (int n = 0; n < nPulse + 1; n++)
    {
      double an = timeWithEnds(n+1) - timeWithEnds(n);
      double phase = angfreqCutoff * an;
      sum1 += x_Si(phase) + std::cos(phase) - 1;
      sum3 += an * an * (-gsl_sf_Ci(phase) + 0.5 * std::pow(sinc(an * freqCutoff), 2) + sinc(2 * an * freqCutoff));
    }

    for (int i = 0; i < nPulse; i++)
    {
      for (int j = i+1; j < nPulse + 1; j++)
      {
        double sign = std::pow(-1.0, i+j);
	// Define time differences for the calculation
        double phi1 = angfreqCutoff * (timeWithEnds(j) - timeWithEnds(i+1));
        double phi2 = angfreqCutoff * (timeWithEnds(j+1) - timeWithEnds(i));
        double phi3 = angfreqCutoff * (timeWithEnds(j) - timeWithEnds(i));
        double phi4 = angfreqCutoff * (timeWithEnds(j+1) - timeWithEnds(i+1));

        sum2 += sign *
                (x_Si(phi1) + x_Si(phi2)
                - x_Si(phi3) - x_Si(phi4)
                + std::cos(phi1) + std::cos(phi2) - std::cos(phi3) - std::cos(phi4));
        sum4 += sign *
                ( - xSq_Ci(phi1) - xSq_Ci(phi2)
                + xSq_Ci(phi3) + xSq_Ci(phi4)
                + phi1 * std::sin(phi1) + phi2 * std::sin(phi2)
                - phi3 * std::sin(phi3) - phi4 * std::sin(phi4)
                - std::cos(phi1) - std::cos(phi2) + std::cos(phi3) + std::cos(phi4));
	
      }
    }
    return S0 * ((sum1 + sum2) / (M_PI * angfreqCutoff) + sum3 + sum4 / angfreqCutoff / angfreqCutoff);
  }
  
  double chi() const
  {
    double overlap;
    if (useFFT == true)
    {
      VectorXd fOmegaAbs2 = computeFOmegaAbs2();
      overlap =  (fOmegaAbs2.head(cutoffIdx).transpose() * sOmega.head(cutoffIdx))(0) * dFreq; // Select 0 element because technically result of vector product is vector of length 1
    }
    else if (noise == "1_over_f_with_low_cutoff")
    {
      double amplitude = 1;
      double cutoff = 1.0 / maxTime;
      overlap = chi_1_over_f_low_cutoff(amplitude, cutoff);
      
      //VectorXd fOmegaAbs2 = computeFOmegaAbs2();
      //overlap =  (fOmegaAbs2.transpose() * weightedSOmega)(0); // Select 0 element because technically result of vector product is vector of length 1

      // Testing:
      double overlap2 = 0;
      VectorXd fOmegaAbs2 = computeFOmegaAbs2();
      overlap2 =  (fOmegaAbs2.transpose() * weightedSOmega)(0); // Select 0 element because technically result of vector product is vector of length 1
      std::cout << std::endl << "Exact chi:" << std::endl << overlap << std::endl;
      std::cout << "Numerical chi:" << std::endl << overlap2 << std::endl;
      std::cout << "Percent error:" << std::endl << (overlap - overlap2) / overlap << std::endl;
    }
    else
    {
      VectorXd fOmegaAbs2 = computeFOmegaAbs2();
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
    return fOmegaAbs2.transpose() * sOmega;
  }

 VectorXd getFilter() const
 {
	 return computeFOmegaAbs2();
 }

};

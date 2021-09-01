# Stochastic Methods of Energy Markets - FinalProject



- **Historical calibration.** Assume that the log-returns of the German power calendar 2021 (the contract that delivers electricity every hour in 2021) are driven by a Variance Gamma process as described in slide 17. After appropriate pre-processing, present a summary of the statistics of the time-series. After performing a ACF plot, calibrate the parameters of the process using the maximum-likelihood method based on the transition density in Equation (1), slide 18. Consider the time-series up to 19th November 2019. Finally do a qqplot and discuss your results.

  The data set is saved in the file Historical Prices FWD Germany.csv, should
   you have MATLAB, you could also read from the file Historical Prices FWD Germany.mat. Note that the files contain price values and not log-returns. Hint You can have
   a look at the files main MLE.m and VGDensity.m

- **Risk Neutral Calibration.** Calibrate the parameters of the Variance Gamma
   process from the quoted options as of 19th November 2019. Such values are
   saved in the file Options Prices Calendar 2021.csv or Options Prices Calendar 2021.mat. Note that these two files contain information relative to several products, you
   have to do some pre-processing and have to filter for DEBY 2021.01, DE stands
   for Germany, B for baseload and Y for year, hence it is the calendar 2021
   for Germany. The underlying calendar settlement prices are saved in For-
   ward Prices.csv or Forward Prices.mat. For the calibration, use either a Monte Carlo-based pricing or implement Equation (5) slide 30. It coincides with the integral formula what we found in one lecture. Note that in this slide it should
   be Black76 model and not Black-Scholes because we are consider a forward
   market. Hint. You can have a look at the files main VG.m, CalibFunction.m
   and VG simulation.m. If helpful, you can also consult the other MATLAB files.

- Compare the results of the two types of calibrations, discuss your findings and plot the differences between the values of the quoted options and the values estimated with the calibrated parameters. Hint Reproduce the plots in slide 38. 


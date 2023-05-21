# Blind Source Seperation Algorithms

This repo is created for a thesis study. In the thesis, different blind source separation (BSS) methods were compared. 7 different BSS methods were used.

* FastICA
* Non-negative Matrix Factorization
* Degenerate Unmixing Estimation Technique
* Spleeter
* Open Unmix
* Wave-U-Net
* Hybrid Demucs

Musdb-HQ database had been used at the experiments. The 4 different components were mixed artifically and demixed after that.

Pre-trained models were used at machine-learning based methods and these implementations were used for classical algorithms as follows;

Non-Negative Matrix Factorization: https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/#Python
FastICA: https://towardsdatascience.com/separating-mixed-signals-with-independent-component-analysis-38205188f2f4
DUET: https://github.com/nussl/nussl

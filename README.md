# Noise-Spectroscopy-ML-project
Time-dependent noise spectroscopy of IBM superconducting qubits using deep learning to estimate and suppress the dephasing errors through optimal dynamical decoupling protocol from SciPy optimiser.

# Noise Spectroscopy using ML

## To evaluate a decoherence curve from an underlying noise spectrum
Siddharth Dhomkar, Department of Physics, IIT Madras

**Based on**:

D.F. Wise, J.J.L. Morton, and S. Dhomkar, “Using deep learning to understand and mitigate the qubit noise environment”, PRX Quantum 2, 010316 (2021)

*   https://doi.org/10.1103/PRXQuantum.2.010316
*   http://dx.doi.org/10.1103/PhysRevApplied.18.024004

## Functions to generate coherence curve from a given set of noise spectra

It is virtually impossible to extract the underlying noise spectrum from a decoherence curve. However, it is possible to obtain an unique coherence decay, if the precise functional form of the noise spectrum is known. This observation is utilized to generate the training datasets.

Qubit noise spectra, $S(\omega)$, can be of various functional forms including $1/f$-like, Lorentzian-like, double-Lorentzian-like etc. It is possible to simulate such noise spectra and corresponding decoherence curve, $C(t)$, can then be evaluated consequently using following relation:

$$- ln C(t) =  \int_{0}^{\infty}\frac{d\omega}{\pi}S(\omega)\frac{F(\omega t)}{\omega^{2}}$$

where, $F(\omega t)$ is the filter function associated with the decoupling protocol used for probing the decoherence dynamics.

(The detailed explanation of data generation process can be found in the manuscript.)

**Generate the filter function**

References pertaining to superconducting qubit noise spectra:

*   https://doi.org/10.1038/ncomms1856
*   https://www.nature.com/articles/ncomms12964
*   https://doi.org/10.1103/PhysRevApplied.18.044026
*   https://doi.org/10.1103/PhysRevResearch.3.013045
*   http://dx.doi.org/10.1103/PhysRevB.79.094520
*   http://dx.doi.org/10.1103/PhysRevB.77.174509
*   https://doi.org/10.1063/1.5089550


[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)]()
# Hierarchical Stochastic Differential Equation Models for Latent Manifold Learning in Neural Time Series

## Table of Contents
* [General Information](#general-information)
* [Reference](#reference)
* [Getting Started](#getting-started)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
<br/>

## General Information
This repository contains the implementation of Hierarchical Stochastic Differential Equation (SDE) Models for latent manifold learning in high-dimensional neural time series. The proposed framework models the latent dynamics using Brownian bridge SDEs, guided by sparse, structured samples from a marked point process. These latent trajectories drive a second set of SDEs that generate observed neural activity, enabling continuous, interpretable, and computationally efficient inference of complex time series. The approach scales linearly with time series length and effectively captures the low-dimensional manifold structure underlying neural recordings.

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2025](). Cite this paper using its [DOI]().

## Getting Started
1. Clone the Repository 
`git clone https://github.com/MaryamOstadsharif/RISE-iEEG.git`

2. Install the required dependencies. `pip install -r requirements.txt`

3. Prepare your dataset

4. Create the `./configs/settings.yaml` according to `./cinfigs/settings_sample.yaml`

5. Create the `./configs/device_path.yaml` according to `./cinfigs/device_path_sample.yaml`

6. Run the `main.py` script to execute the model.

## Repository Structure
This repository is organized as follows:

- `/main.py`: The main script to run.

- `/src/preprocessing`: Contains scripts for data loading and preprocessing.

- `/src/experiments/`: Contains scripts for different experiments.
  
- `/src/model/`: Contains the functions required to build, train and evaluate models.

- `/src/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `/src/utils`: Contains utility script `utils.py` for some helper function that are used in code.

- `/src/interpretation`: Contains script to run the Integrated Gradients method for model interpretation.
  
- `src/visualization`: Contains the `vizualize_utils.py` script for data and result visualization.
<br/>

## Citations
The code contained in this repository for RISE-iEEG is companion to the paper:  


which should be cited for academic use of this code.  
<br/>

## Contributing

We encourage you to contribute to RISE-iEEG! 

## License

This project is licensed under the terms of the MIT license.

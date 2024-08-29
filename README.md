# hep-aid
**_hep._** High energy physics phenomenology.

**_aid._** Help, typically of a practical nature.

~*aid.* **A**rtificial **I**ntelligence **D**evelopment~

## Overview

In this work, we introduce a new Python library, `hep-aid`, which provides a modular framework for developing parameter scan algorithms for Beyond Standard Model (BSM) phenomenology. It manages High Energy Physics (HEP) software and provides components to ease the utilization, implementation, and development of parameter scan algorithms for phenomenological studies. The library comprises two main modules: the `hep` module and the `search` module.

## Modules

### `hep` Module

- **Purpose**: Facilitates the integration of the HEP software ecosystem into Python scripts.
- **Features**:
  - Allows users to perform a numerical point-wise evaluation of a parameter space point across a series of HEP software, wich we called HEP-Stacks.
  - Currently, the SARAH family of programs is implemented: First generate the BSM model files with [SARAH](https://sarah.hepforge.org/). Then, [SPheno](https://spheno.hepforge.org/), [HiggsBounds](https://higgsbounds.hepforge.org/), [HiggsSignals](https://higgsbounds.hepforge.org/) and [MadGraph](https://launchpad.net/mg5amcnlo) can be run individually or sequentially as a HEP-Stack in `hep-aid`.

### `search` Module

- **Purpose**: Manages Parameter Scans algorithms using an Active Search Paradigm.
- **Features**:
  - Utilises a search policy and a surrogate model to explore parameter spaces.
  - Supports various PS methods, including Markov Chain Monte Carlo (MCMC) and machine learning (ML) based sampling methods.
  - Includes an objective function constructor to define search space, objectives, and constraints based on a predefined configuration.
  - Maintains an internal dataset of samples.
  - Provides functionalities for saving, loading, and exporting datasets in formats such as Pandas DataFrame or PyTorch tensor.

### `search.objective` 

- The connection between the Parameter Scans algorithms in the `search` module and the HEP-Stacks in the `hep` module is established through the construction of an **objective function**. The `search` module includes an objective function constructor, which defines the search space, objectives, and constraints based on a predefined configuration.

## Parameter Scan Algorithms

This library was originally develop for experimentation for developing the [b-CASTOR](https://arxiv.org/abs/2404.18653) parameter scan method. Where a surrogate model to approximate the BSM model outputs and a search policy funtions needs to be defined. However, many Parameter Scans algorithms can be implemented following this paradigm and the current objectie of `hep-aid` is to ease the development, implementation and utilisation of general parameter scans algorithms. `hep-aid` currently includes:
  - 

## Installation

To install the hep-aid library clone this repository, an install by:
```bash
pip install .
```
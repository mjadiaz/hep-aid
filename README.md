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
  - Allows users to perform a numerical point-wise evaluation of a parameter space point across a series of HEP software.
  - Collects output with a single function call.
  - Currently supports the SARAH family of programs.

### `search` Module

- **Purpose**: Manages Parameter Space (PS) algorithms using an active search paradigm.
- **Features**:
  - Utilizes a search policy and a surrogate model to explore parameter spaces.
  - Supports various PS methods, including Markov Chain Monte Carlo (MCMC) and machine learning (ML) based sampling methods.
  - Includes an objective function constructor to define search space, objectives, and constraints based on a predefined configuration.
  - Maintains an internal dataset of samples.
  - Provides functionalities for saving, loading, and exporting datasets in formats such as Pandas DataFrame or PyTorch tensor.

## Integration

The connection between the PS algorithms in the `search` module and the HEP software in the `hep` module is established through the construction of an **objective function**. The `search` module includes an objective function constructor, which defines the search space, objectives, and constraints based on a predefined configuration.

## Installation

To install the hep-aid library clone this repository, an install by:
```bash
pip install .
```
if you want to modify the library you can install in developer mode by:
```bash
pip install -e .
```
## Overview

AMUSE (Adaptive Mutation Strategy for Unsupervised Feature Selection) is a novel differential evolutionary algorithm designed to enhance the feature selection process for large datasets. It integrates an adaptive mixed mutation strategy for generating diverse mutation spaces and an assistance mechanism for selecting the optimal crossover probability ($CR$) during the crossover process.

### Key Features:

- Adaptive Mixed Mutation Strategy: Dynamically generates mixed mutation spaces to enhance diversity and performance.
- Assistance Mechanism: Automatically selects the optimal crossover probability ($CR$) based on the standard deviation along the individual dimension.
- Improved Mutation Strategies: Incorporates 'local best' to reduce randomness and combines with existing mutation strategies to form a pool.
- Fitness Function: Integrates multiple indicators for effective unsupervised feature selection.

## Dependencies

- Python: 3.8
- gensim: 0.13.4
- gym: 0.26.2
- gymnasium: 0.29.1
- numpy: 1.22.4
- scikit-learn: 1.3.2

## Usage Example

### Hypothyroid Dataset

The `main_hypothyroid.py` file serves as the main entry point for running AMUSE on the hypothyroid dataset.

#### Steps to Run:

1. Ensure all dependencies are installed.
2. Place the hypothyroid dataset in the appropriate directory.
3. Run `main_hypothyroid.py` using the command:

```bash
python main_hypothyroid.py
```

This script will execute the AMUSE algorithm on the hypothyroid dataset, showcasing its feature selection capabilities.
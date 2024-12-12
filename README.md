# README

## Label Noise: Tradeoff Between Fairness and Performance

Folder (experiments/E1) contains implementation of the **2LR Plugin Approach** (The cost of fairness in binary classification) to analyze how label noise affects the tradeoff between fairness and performance in machine learning models. Experiment E1 evaluates:

- **Fairness Metric:** Mean Difference (MD)
- **Performance Metric:** Balanced Error Rate (BER)

The analysis utilizes synthetic datasets with configurable label noise(flipping probability). The code for the sysnthetic dataset is copied from Zafar et al. 2017 - Fairness Constraints: Mechanisms for Fair Classification


### Repository Structure

- **`tools/gen_synth_data.py`**: Utility functions to generate synthetic datasets.
- **`tools/corrupt_labels.py`**: Functions to add noise to labels.
- **`tools/calc_metrics.py`**: Implementations of performance and fairness metrics (e.g., BER, MD, DI).
- **`tools/plot_helper.py`**: Visualization helpers for plotting.
- **Main Experiments**: Executes the experiments and generates plots comparing the effects of label noise.
- **`experiments/E1 - Label Noise/`**: Code for 2LR Plugin Apprach with label noise .

---

### Getting Started

#### Prerequisites

- Python >= 3.8
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Install the dependencies using:

```bash
pip install numpy matplotlib scikit-learn
```

#### Running the Experiment

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the jupyter notebook in E1 to reproduce results:

3. Generated plots will be saved in the `img/E1 - Label Noise/` directory.

---

### Key Features

- **Synthetic Data Generation**: Simulate data with specified levels of disparity and noise. Disparity controls relation of sensitive attribute and input X
- **Noise Injection**: Add binary label noise with a configurable probability.
- **Metric Computation**: Measure fairness and performance using:
  - Balanced Error Rate (BER)
  - Mean Difference (MD)
  - Disparate Impact (DI)
- **Visualization**: Generate plots to visualize tradeoffs and trends.

---

### Implementation Details

#### Data Generation
Synthetic data is generated with adjustable parameters, such as the degree of disparity (`disc_factor`) and number of samples. The labels and sensitive attributes can be independently corrupted using `add_bin_noise`.

#### Model Training
Two logistic regression models are trained:

- **Performance Model (P):** Predicts class labels.
- **Fairness Model (F):** Predicts sensitive attribute labels.

These models are combined using the 2LR plugin approach to evaluate the tradeoff between fairness and performance.

#### Fairness-Performance Tradeoff
A range of lambda (λ) values is evaluated to quantify the impact of fairness on performance:

- **Low λ:** Performance prioritized over fairness.
- **High λ:** Fairness prioritized over performance.

---

### Outputs

1. **Plots:**
   - BER vs. MD (Mean Difference)
   - BER vs. DI (Disparate Impact)
   - Effects of label noise on BER

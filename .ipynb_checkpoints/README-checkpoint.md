# FULDebug - Federated Unlearning Debugging Framework

This repository contains the artifacts for **FULDebug**, a federated unlearning debugging framework developed as part of our research. FULDebug enables interactive analysis and debugging during the federated learning process by offering breakpoints, step-in, and real-time telemetry-based analytics.

| ![Debug Before Unlearning](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Gifs/preunlearning.gif?raw=true) | ![Debug During Unlearning](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Gifs/during-unlearning.gif?raw=true) | ![Visualize Metrics](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Gifs/plotting.gif?raw=true) |
|:-----------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------:|
| How to debug before unlearning                              | How to debug during the unlearning                               | Visualize metrics                                 |

## How to Run the Project

### 1. Setup Python Environment
To get started, ensure that you have Python installed on your system. The required dependencies can be installed by running:

```bash
pip install -r requirements.txt
```

### 2. Running FULDebug
Once the environment is set up, you can run the **FULDebug** framework by executing the following command:

```bash
python3 ful-debug.py
```

### 3. Configure the Federated Learning Setup
When running the framework, you'll be prompted to input the number of clients and the number of federated learning (FL) rounds. These parameters will help set up the federated learning environment for your experiment.

### 4. Interactive CLI Debugging
After the FL setup, follow the instructions in the CLI to analyze various metrics and perform debugging. You can:
- View analytics at different stages of the unlearning process.
- Press `Ctrl + C` at any point to set a **breakpoint** in the current FL round, which allows you to step into, step out of, or resume the unlearning analysis.

### 5. Other Python Files
The remaining Python files in the project are related to the core framework and unlearning debugging mechanisms.

### 6. Jupyter Notebooks
The repository also includes `.ipynb` files for running **Jupyter Notebooks** that allow you to visualize and analyze the unlearning process more easily. These notebooks provide a more interactive approach to exploring the FULDebug framework’s results and insights.

## Project Structure

- **ful-debug.py**: Main script to run the FULDebug framework.
- **requirements.txt**: File containing the Python dependencies.
- **framework/analytics.py**: Core analytics engine that provides the client vs global model comparison, client contributions, and unlearning evaluation methods.
- **FUL_Algo**: Contains methods for performing unlearning operations and applying PGD for robustness.
- **framework/debugger_interface.py**: CLI-based interactive debugger interface that allows step-in, step-out, and telemetry analysis during unlearning.

### Directory Structure

- **Results**: Contains the output files for each experiment, including captured results during federated learning and unlearning.
- **Results/images**: Contains the plots and visualizations related to the evaluation of the framework, generated during runtime.


## Evaluation & Results
The **Results** directory stores all the captured results, including accuracy, contribution analysis, and other key metrics. Inside the **Results/images** subdirectory, you'll find the plots related to various evaluations such as:
- Pre and post-unlearning accuracy.
- Client contribution analysis.
- Class-wise accuracy comparisons.

These artifacts demonstrate the effectiveness of FULDebug in managing and debugging the federated unlearning process.

| ![Class Balancing](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Images/class_wise_analysis.png?raw=true) | ![Overall Accuracy](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Images/overall_accuracy_comparison.png?raw=true?raw=true) | ![Runtime Overhead](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Images/runtime_overhead.png?raw=true?raw=true) |
|:-----------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------:|
| Class Balancing                              | Overall Accuracy                               | Runtime Overhead                                 |

---

## Other Metrices
Our framework provides a range of key metrics and visualizations to verify the unlearning process and identify the impacts of unlearning specific clients. These include class-wise accuracy comparisons, feature importance deviations, and SHAP-based visual explanations. These metrics allow analysis of how client contributions affect model performance and feature significance. Inside the **Results/OtherMetrics** subdirectory, you'll find the plots related to various evaluations such as:

| ![Feature Deviation](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Other-Key-Metrices/feature_importance_deviation.jpeg?raw=true) | ![Debug During Unlearning](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Other-Key-Metrices/shap_analysis.jpeg?raw=true) | ![Feature Change Score](https://github.com/FULDebug/FULDebug-Artifacts/blob/main/Result/Other-Key-Metrices/feature_score_between_global_model_before_and_after.jpeg?raw=true) |
|:-----------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------:|
| Feature Deviation                              | Debug During Unlearning                               | Feature Change Score                                 |


Feel free to explore the framework by running the scripts and notebooks to see the real-time analysis and evaluation of federated unlearning.
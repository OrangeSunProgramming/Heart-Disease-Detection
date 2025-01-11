# Heart Disease Detection Using CNN

This project aims to develop a Convolutional Neural Network (CNN) for the multi-label classification of heart disease using the BMD-HS dataset. The goal is to identify positive cases of heart disease with high recall while maintaining a balance with precision.

## Dataset
The BMD-HS dataset contains over 800 heart sound recordings classified into six categories, including common valvular diseases and healthy samples. The dataset includes multi-label annotations, echocardiographic data, and rich metadata.

## Key Features
- **Multi-label annotations**: Capture unique disease states.
- **Echocardiographic data**: Provide additional diagnostic context.
- **Diverse demographic representation**: Gender-balanced collection.
- **Balanced class representation**: Address class imbalance issues.
- **Rich metadata**: Enable in-depth research and potential discovery of new correlations.
- **Multi-disease data**: Reflect real-world scenarios with multiple valvular diseases.

## Project Structure
- **data**: Contains the dataset files.
- **images**: Contains images of feature maps and threshold graphs.
- **src**: Contains the source code files.

## Methodology

### Data Preparation
- **Preprocessing**: Applied various preprocessing techniques to clean and normalize the data.
- **Data Augmentation**: Initially applied data augmentation techniques but found that it introduced noise, leading to suboptimal results. Therefore, data augmentation was not used in the final model.

### Model Architecture
- **CNN Model**: Developed a CNN model optimized using the Optuna TPE algorithm.
- **Hyperparameter Tuning**: Used Optuna's TPE algorithm to maximize the F1 score, balancing recall and precision.
- **Early Stopping**: Implemented early stopping with a patience of 20 to monitor validation loss and restore the best weights.

### Training
- **Class Imbalance Handling**: Used sample weights to balance the dataset during training.
- **Evaluation Metrics**: Calculated weighted F1 score during training and macro F1 score after applying sample weights.

### Custom Metrics
Due to the deprecation of TensorFlow Addons (TFA), I created custom implementations for the F1 score and Hamming loss. These custom metrics were essential for evaluating the model's performance in a multi-label classification setting.

### Evaluation
- **Threshold Optimization**: Found the best threshold to maximize the F1 score, balancing recall and precision.

## Results

| Metric       | Value Before Threshold Optimization | Value After Threshold Optimization |
|--------------|-------------------------------------|------------------------------------|
| **AUC**      | 0.8166                              | 0.8166                             |
| **F1 Score** | 0.6773                              | 0.6914                             |
| **Loss**     | 0.4689                              | 0.4689                             |
| **Precision**| 0.5872                              | 0.6039                             |
| **Recall**   | 0.7678                              | 0.8184                             |
| **Threshold**| -                                   | 0.45                               |

## Model Performance Metrics

| Metric       | Value  | Description                                                                                 |
|--------------|--------|---------------------------------------------------------------------------------------------|
| **AUC**      | 0.8166 | Measures the ability to distinguish between classes. Higher values indicate better performance. |
| **F1 Score** | 0.6914 | Harmonic mean of precision and recall, reflecting a balance between identifying positives and minimizing false positives. |
| **Loss**     | 0.4689 | Measures error in predictions. Lower loss indicates better performance.                      |
| **Precision**| 0.6039 | Proportion of true positive predictions among all positive predictions.                     |
| **Recall**   | 0.8184 | Proportion of true positive predictions among all actual positive cases.                    |
| **Threshold**| 0.45   | Probability cutoff for classifying cases as positive, balancing precision and recall.        |

### Interpretation of Metrics
- **High Recall**: Crucial for identifying positive cases of heart disease, with recall at 81.84%.
- **Balanced Precision**: Acceptable level of 60.39%, minimizing false negatives in diagnostics.
- **Optimal F1 Score**: A score of 0.6914 reflects a strong balance between precision and recall.
- **AUC**: Indicates the model's ability to distinguish between heart disease and healthy cases effectively.

### Visualizations
Below are visualizations that illustrate the model's performance:

- **Precision vs Threshold**
- **Recall vs Threshold**
- **F1 Score vs Threshold**

## Files

| File Name                  | Description                                     |
|----------------------------|-------------------------------------------------|
| **HyperPTuning.py**        | Hyperparameter tuning using Optuna.             |
| **cnn_model.py**           | CNN model architecture.                         |
| **custom_f1_score.py**     | Custom F1 score implementation.                 |
| **custom_hamming_loss.py** | Custom Hamming loss implementation.             |
| **data_augmentation.py**   | Data augmentation techniques.                   |
| **data_preparation.py**    | Data preparation and preprocessing.             |
| **model_info.py**          | Model information and summary.                  |
| **optuna_best_params.py**  | Best hyperparameters from Optuna.               |
| **sample_weights.py**      | Calculation of sample weights.                  |

## Conclusion
This project demonstrates the application of CNNs for heart disease detection with a focus on maximizing recall while balancing precision. The use of Optuna for hyperparameter tuning and sample weights for handling class imbalance were key to achieving the results.

## Requirements
- Python 3.17
- TensorFlow
- Optuna
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

## Acknowledgements
This project uses the BMD-HS dataset. If this dataset helped your research, please cite the following paper:

Ali, S. N., et al. (2024). BUET Multi-disease Heart Sound Dataset: A Comprehensive Auscultation Dataset for Developing Computer-Aided Diagnostic Systems. arXiv preprint arXiv:2409.00724.

## Installation
```bash
pip install -r requirements.txt

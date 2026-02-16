# Spacecraft Anomaly Challenge
Using a Temporal Convolutional Network for anomaly detection in sensor measurements collected from a large spacecraft.

## Temporal Convolutional Network Anomaly Detector - Python
### Resource - Kaggle
This project utilizes sensor spacecraft data measured by the European Space Agency (ESA). This dataset has been made publically available via Kaggle for an anomaly detection challenge. Visit [here](https://www.kaggle.com/competitions/esa-adb-challenge/overview) for data resource and more information.
### Method
The data is first loaded into memory using Pandas. Target channels for tracking are also imported. Data preprocessing steps are conducted.
A train-validation split is created (80:20 split) and the feature values are scaled using Scikit-Learn's RobustScaler for proper scaling with anomalous data.
A PyTorch dataset bins the time-series data into sliding windows for model training. After binning, batch size is declared and the processed dataset is created.
Next, I build the baseline model. This model features 2 convolutional layers and 1 fully connected layer for classification. When configuring the model, I use the Adam optimizer and a binary cross-entropy with logits loss as the criterion. At this point I use the pos_weight argument to address class imbalance within the data. In the training loop, I use the F-0.5 evaluation metric to score the validation dataset. The metric takes the following form:


$$F_{0.5} = \frac{(1+0.5^2)*Precision_{e_{corr}}*Recall_e}{0.5^2*Precision_{e_{corr}}+Recall_e}$$


The goal of the model is to minimize false positives. 

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) Pandas
2) PyTorch
3) Scikit-Learn
4) Matplotlib
5) Numpy

The notebook was tested using Python 3.10.11.

### Pre-requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

    
Note: Dataset and the specified directories within the notebook are NOT included in this repository due to file size concerns; refer to the hyperlink under the "Resource" section at the beginning of the README file for access to the dataset.
An integrated development environment (IDE) is necessary to run this notebook. 

### Version History
V1.0 - The Jupyter Notebook is created. Data processing pipeline is drafted and functional. 

V1.1 - EDA is performed.

V1.2 - Changed the model API from TensorFlow to PyTorch to build a TCN over an LSTM. The baseline model is drafted. 


## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook. Furthermore, the outputs of the program are the following:
1) Parquet file: Model outputs for the test dataset are compressed to the parquet file format when saved onto local machine.


### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset.

Using the notebook:
1) Open the notebook within IDE.
2) Run all cells in the notebook.
3) Validate process externally by checking if the model outputs for the test set were saved locally.
4) The model outputs are ready to be validated and model performance evaluated.
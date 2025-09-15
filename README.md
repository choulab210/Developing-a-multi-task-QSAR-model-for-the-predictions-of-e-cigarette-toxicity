# -*- coding: utf-8 -*-

# Developing a multi-task QSAR model for the prediction of e-cigarette toxicity
This repository contains all raw data and Python code files for the project entitled  
**"Developing a multi-task QSAR model for the prediction of e-cigarette emissions toxicity"**

## Respository Structure
Within the **`Code`** folder there are 2 Subfolders:

### 1. `Training & Evaluation Models`
This subfolder contains local scripts (.py) and Google Colab Notebooks (.ipynb) for the development of:
- Single-Task QSAR Models (`Single-Task_QSAR`)
- Multi-Task Eye & Skin toxicity Models (`ESS_Multi-Task_Train`)
- Multi-Task Carcinogencitiy & Genotoxicity Models (`CancerGenotoxicity_Multi-Task`)
   
Performance metrics including *accuracy*, *balanced accuracy*, *MCC*,  *precision*, *recall*, *F1-Score*, and *ROC-AUC* are reported for comparative analysis for all models. All models also include additional analyses for interpretation including Applicability Domain plots, Confusion Matricies, and Feature Importances. 
   The dataset used to train and evaluate these models can be found in the `Datasets`subfolder.

### 2. `User Predict`
This subfolder contains local scripts (.py) and Google Colab Notebooks (.ipynb) designed for making prediction on new compounds using the pre-trained models from the `Training & Evaluation Models` folder.
The scripts/notebooks allows users to input new SMILES and obtain task-specific toxicity predictions and confidence for:
- Multi-Task Eye & Skin toxicity Models (`ESS_Model_Predict`)
- Multi-Task Carcinogencitiy & Genotoxicity Models (`CarGen_Model_Predict`)
    
The pre-trained model files (.pth) and, if applicable, descriptor scaler files (.pkl) can be found in the **`Trained Model & Hyperparameters`** folder.

## Contributors

- **Author:** Alexa Canchola, University of California, Riverside
- **Author:** Kunpeng Chen, University of California, Riverside
- **Advisor:** Wei-Chun Chou, University of California, Riverside  


*The manuscript describing the files in this repository is currently under review.*

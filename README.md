# Interpretable Multi-Modal Artificial Intelligence Model for Predicting Locally Advanced Gastric Cancer Response to Neoadjuvant Chemotherapy

## Requirement
These codes can run on Windows 11 with Python 3.7,R-4.3.1 and CUDA (GPU computation).

### Create a Python code dependency environment by performing the following steps.
```bash
conda env create -n iSCLM_demo --file env.yml
conda activate iSCLM_demo
```

## Performance computation
Files within `Performance computation` folder are used to calculate model data, such as AUC and ACC, using the `countAUC.R` and `countACC.R` files, respectively.

## Code
The provided codes are involved in model prediction and the validation process for Fig 1, Supplementary Table 4, and Supplementary Table 5.

### Model prediction
The `predict` folder contains CT model prediction codes and pathology model prediction codes, which can be used separately to output prediction values.

### Fig. 5
The `SHAP.py` file implements the visualization of sample SHAP values and outputs the SHAP value matrix. To achieve this, we have modified functions within the shap package, and it is necessary to replace the `_colors.py` and `_image.py` files in the corresponding locations within the shap package.

### Supplementary Table 4
Use the `STEP1 Mean absolute deviation calculation.py` file to process and calculate the patch distance values, and then use the `STEP2 Non-parametric test for patch distance.R` file to verify the distance values.

### Supplementary Table 5
Use the `Cellular component t-test.R` file to process and calculate the cell data, output the file, and perform further verification.

# Interpretable multi-modal artificial intelligence model for predicting locally advanced gastric cancer response to neoadjuvant chemotherapy

## Requirement
This code has been tested on Windows 11, version 23H2 with Python 3.9.7 and  CUDA (GPU computation)

### Create a python code dependency environment by performing the following steps.

```bash
conda env create -n iSCLM_demo --file env.yml
conda activate iSCLM_demo
```

##Demo
The demo file is used to calculate model data, such as AUC and ACC, using the countAUC.R and countACC.R files, respectively.

##Code
The provided codes are involved in model prediction and the validation process for Fig 5, Supplementary Table 5, and Supplementary Table 6.

###Predict
The Predict folder contains CT model prediction code and pathology model prediction code, which can be used separately to output prediction values.

###Fig. 5
The SHAP.py file implements the visualization of sample SHAP values and outputs the SHAP value matrix. To achieve this, we have modified functions within the shap package, and it is necessary to replace the _colors.py and _image.py files in the corresponding locations within the shap package.

###Supplementary Table. 5
Use the STEP1.py file to process and calculate the patch distance values, and then use the STEP2.R file to verify the distance values.

###Supplementary Table. 6
Use the Supplementary Table. 6.R file to process and calculate the cell data, output the file, and perform further verification.

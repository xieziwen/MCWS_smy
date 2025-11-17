# Corn Jointing Stage CH (Plant Height) Inversion Model (R Implementation)


## Project Overview
This project focuses on inverting the **Plant Height (CH)** of corn at the jointing stage using spectral index data. It leverages R language to implement a complete workflow, including data preprocessing (PCA dimensionality reduction), multi-model training, performance evaluation, and full-dataset prediction. By comparing five machine learning algorithms—Support Vector Machine (SVM), Random Forest (RF), Gaussian Process Regression (GSL), Partial Least Squares (PLS), and Neural Network (NN)—the script automatically selects the optimal model for CH inversion. This tool provides precise and efficient technical support for corn growth monitoring in agricultural research.


## Environment Configuration
### Dependent Packages Installation
Run the following code in the R environment to install all required packages:
```r
# Install core dependent packages for model training and data processing
install.packages(c(
  "pls",        # Partial Least Squares Regression
  "car",        # Companion to Applied Regression (for statistical utilities)
  "e1071",      # Support Vector Machines & Naive Bayes algorithms
  "caret",      # Classification And Regression Training (model tuning)
  "glmnet",     # Lasso/Elastic-Net regularization (for model optimization)
  "gbm",        # Gradient Boosting Machines (optional for extended models)
  "tidyverse",  # Data wrangling (dplyr) and visualization (ggplot2)
  "kernlab",    # Kernel-based machine learning (supports GSL)
  "rio"         # Data import/export (supports Excel .xlsx format)
))
```

### System Requirements
- **R Version**: ≥ 4.0.0 (for compatibility with modern R packages)
- **Operating System**: Windows / macOS / Linux  
  *Note: Adjust working directory paths for different systems (e.g., `E:/path` for Windows, `/home/username/path` for Linux/macOS).*


## Data Description
### Input Data Specifications
| Item                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Format              | Excel file (`.xlsx`); primary data sheet: `Sheet 1`                         |
| Core Variables      | - **Target Variable**: `CH` (Corn Plant Height, unit: cm)<br>- **Features**: SPAD (chlorophyll content), LAI (Leaf Area Index), LNC (Leaf Nitrogen Content), spectral bands (blue, green, red, rededge, near-infrared/nir), and vegetation indices (NDVI, GNDVI, GRVI, etc.) |
| Example Data File   | `数据汇总-拔节期.xlsx` (Corn Jointing Stage Data Summary)                     |
| Required Fields     | Must include the `CH` column and at least 3 spectral/vegetation index features to ensure model stability |

### Data Preprocessing Workflow
1. **Data Splitting**:  
   Randomly splits data into a **training set (67%)** and **test set (33%)** using `set.seed(247)` to ensure result reproducibility.  
2. **Factor Variable Handling**:  
   Converts categorical features to dummy variables and aligns column names between training and test sets to avoid dimension mismatch during model prediction.  
3. **PCA Dimensionality Reduction**:  
   Performs Principal Component Analysis (PCA) on features. Retains principal components (PCs) with **cumulative variance ≥ 95%** to reduce feature redundancy and avoid overfitting.


## Usage Steps
### 1. Clone the Repository
Run the following commands in Terminal (Git Bash / Command Prompt) to clone the repository to your local machine:
```bash
# Clone the repository
git clone https://github.com/[Your-GitHub-Username]/corn-ch-inversion.git

# Navigate to the project directory
cd corn-ch-inversion
```

### 2. Configure the Working Directory
Modify the `setwd()` function in the R script to point to your local data storage path. Examples for different operating systems:
```r
# For Windows (replace with your actual path)
setwd("E:/1_wzy/corn/predict/data")

# For macOS/Linux (replace with your actual path)
# setwd("/home/username/corn/predict/data")
```

### 3. Run the Script
Execute the script in R or RStudio using one of the following methods:
```r
# Option 1: Source the main script (if saved as "corn_ch_inversion.R")
source("corn_ch_inversion.R")

# Option 2: Run code line-by-line in RStudio (for debugging or customization)
```

### 4. Access Output Results
All results are saved in the `output` folder (automatically created if it does not exist). The output files include:

| Output File                  | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `拔节期-CH模型评估.csv`       | Model performance metrics (R², RMSE, MAE) for both training and test sets across all 5 models. |
| `拔节期-CH反演结果.csv`       | Full-dataset results: actual CH values + predicted values from all models.  |
| RF Visualization Plot        | Scatter plot of "Predicted CH vs. Actual CH" for the Random Forest model, with annotations for R² and RMSE (training/test sets). |


## Model Details
### Built-in Machine Learning Models
| Model Abbreviation | Full Name                  | Key Tuned Parameters                          | Purpose                                      |
|--------------------|----------------------------|-----------------------------------------------|----------------------------------------------|
| SVM                | Support Vector Machine     | `sigma` (10⁻³ to 10¹), `C` (10⁻¹ to 10³)       | Handles non-linear relationships in spectral data |
| RF                 | Random Forest              | `mtry` (1 to total number of features)         | Reduces overfitting; robust to noisy data     |
| GSL                | Gaussian Process Regression| `sigma` (10⁻³ to 10¹)                          | Captures complex patterns with kernel functions |
| PLS                | Partial Least Squares      | `ncomp` (1 to min(10, total features))         | Reduces multicollinearity in high-dimensional data |
| NN                 | Neural Network             | `size` (5 to 25), `decay` (10⁻³ to 10⁰)        | Models non-linear relationships with hidden layers |

### Evaluation Metrics
- **R² (Coefficient of Determination)**: Measures model fit (range: 0–1; higher values indicate better fit).  
- **RMSE (Root Mean Squared Error)**: Quantifies prediction error (unit: cm; lower values indicate better precision).  
- **MAE (Mean Absolute Error)**: Measures average absolute prediction error (unit: cm; lower values indicate more stable predictions).  


## Results Interpretation
1. **Optimal Model Selection**:  
   The script automatically identifies the model with the **smallest test-set RMSE** as the optimal CH inversion model (Random Forest often performs best for spectral data).  
2. **Output File Insights**:  
   - Use `拔节期-CH模型评估.csv` to compare performance across models (e.g., RF typically has higher R² and lower RMSE).  
   - Use `拔节期-CH反演结果.csv` for batch prediction of corn CH in large datasets.  
3. **Visualization Details**:  
   The scatter plot includes:  
   - Blue points (training set) and orange points (test set).  
   - A 45° reference line (represents ideal "predicted = actual" performance).  
   - Text annotations for R² and RMSE (training/test sets) for quick performance assessment.  


## License
This project is licensed under the **MIT License**—see the [LICENSE](LICENSE) file for detailed terms.


## Contribution Guidelines
We welcome contributions to improve the project.


## Contact Information
For questions or collaboration inquiries:
- Email: [smy511@henau.edu.cn]

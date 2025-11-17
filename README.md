# Corn Jointing Stage CH Inversion Model (R Implementation)
# 玉米拔节期株高（CH）反演模型（R语言实现）


## Project Overview / 项目概述
This project focuses on inverting the Plant Height (CH) of corn at the jointing stage using spectral index data. Leveraging R language, it implements a complete workflow including data preprocessing (PCA dimensionality reduction), multi-model training, performance evaluation, and full-dataset prediction. By comparing five machine learning algorithms (SVM, RF, GSL, PLS, NN), the optimal model for CH inversion is automatically selected, providing a precise and efficient tool for corn growth monitoring in agricultural research.

本项目针对玉米拔节期株高（CH，Plant Height）反演需求，基于光谱指数数据（含SPAD、LAI、多波段反射率及植被指数等），使用R语言实现数据预处理（PCA降维）、多模型训练、性能评估与全数据预测的完整流程。通过对比5种机器学习算法（支持向量机、随机森林等），自动筛选最优株高反演模型，为农业研究中玉米生长监测提供精准高效的技术支持。


## Environment Configuration / 环境配置
### Dependent Packages / 依赖包安装
Run the following code in R to install all required packages:  
在R环境中运行以下代码，安装所有依赖包：
```r
# Install core dependent packages
# 安装核心依赖包
install.packages(c(
  "pls",        # Partial Least Squares Regression
  "car",        # Companion to Applied Regression
  "e1071",      # Support Vector Machines & Naive Bayes
  "caret",      # Classification And Regression Training
  "glmnet",     # Lasso & Elastic-Net Regularized Generalized Linear Models
  "gbm",        # Gradient Boosting Machines
  "tidyverse",  # Data Wrangling & Visualization (includes ggplot2)
  "kernlab",    # Kernel-Based Machine Learning
  "rio"         # Data Import/Export (supports Excel files)
))
```

### System Requirements / 系统要求
- **R Version**: ≥ 4.0.0  
- **Operating System**: Windows / macOS / Linux  
  - Note: Adjust the working directory path format for different systems (e.g., `E:/path` for Windows, `/home/path` for Linux/macOS).  
  - 注意：不同系统需适配工作目录路径格式（Windows用`E:/路径`，Linux/macOS用`/home/路径`）。


## Data Description / 数据说明
### Input Data / 输入数据
| Item                | English Description                                  | 中文说明                                          |
|---------------------|------------------------------------------------------|-------------------------------------------------|
| Format              | Excel file (`.xlsx`)                                  | Excel文件（`.xlsx`格式）                          |
| Core Variables      | Target: `CH` (Plant Height, unit: cm); Features: SPAD, LAI, LNC, spectral bands (blue/green/red/rededge/nir), vegetation indices (NDVI, GNDVI, etc.) | 目标变量：`CH`（株高，单位：cm）；特征变量：SPAD、LAI、LNC、光谱波段（蓝/绿/红/红边/近红外）、植被指数（NDVI、GNDVI等） |
| Example Data        | `数据汇总-拔节期.xlsx` (Sheet 1)                      | 示例数据文件：`数据汇总-拔节期.xlsx`（第1个工作表） |
| Required Fields     | Must include `CH` and at least 3 spectral/VI features | 必须包含`CH`列及至少3个光谱/植被指数特征列         |

### Data Preprocessing Workflow / 数据预处理流程
1. **Data Splitting**: Randomly split data into training set (67%) and test set (33%) using `set.seed(247)` for reproducibility.  
   **数据划分**：使用`set.seed(247)`保证可复现性，按67%:33%比例随机分割训练集与测试集。  
2. **Factor Variable Handling**: Convert categorical variables to dummy variables and align column names between training/test sets to avoid mismatch.  
   **因子变量处理**：将分类变量转换为虚拟变量，并对齐训练集与测试集列名，避免维度不匹配。  
3. **PCA Dimensionality Reduction**: Perform PCA on features, retain principal components (PCs) with cumulative variance ≥ 95% to reduce feature redundancy.  
   **PCA降维**：对特征变量执行主成分分析，保留累计方差解释率≥95%的主成分，减少特征冗余。


## Usage Steps / 使用步骤
### 1. Clone the Repository / 克隆仓库
Run the following command in Terminal (Git Bash / Command Prompt):  
在终端（Git Bash/命令提示符）中运行以下命令：
```bash
# Clone the repository to local
# 克隆仓库到本地
git clone https://github.com/[Your-GitHub-Username]/corn-ch-inversion.git

# Enter the project directory
# 进入项目目录
cd corn-ch-inversion
```

### 2. Configure Working Directory / 配置工作目录
Modify the `setwd()` function in the R script to your local data path:  
修改R脚本中的`setwd()`函数，指向本地数据存储路径：
```r
# Example for Windows
# Windows系统示例
setwd("E:/1_wzy/corn/predict/data")

# Example for macOS/Linux
# macOS/Linux系统示例
# setwd("/home/yourname/corn/predict/data")
```

### 3. Run the Script / 运行脚本
```r
# Import the main script (if saved as "corn_ch_inversion.R")
# 导入主脚本（若脚本保存为"corn_ch_inversion.R"）
source("corn_ch_inversion.R")

# Or run the code line-by-line in RStudio
# 或在RStudio中逐行运行代码
```

### 4. View Output Results / 查看输出结果
All results are saved in the `output` folder (automatically created if not exists):  
所有结果保存在`output`文件夹中（不存在则自动创建）：

| Output File                  | English Description                                                                 | 中文说明                                                  |
|------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------|
| `拔节期-CH模型评估.csv`       | Model performance metrics (R², RMSE, MAE) for training/test sets of 5 models.         | 5个模型的训练集/测试集性能指标（R²、RMSE、MAE）          |
| `拔节期-CH反演结果.csv`       | Actual CH values + predicted values of all models for the full dataset.               | 全数据集的实际CH值与所有模型的预测值                      |
| Visualization Plot           | Scatter plot of "Predicted CH vs Actual CH" for RF model (with R²/RMSE annotations).  | 随机森林模型的“预测CH vs 实际CH”散点图（标注R²/RMSE指标） |


## Model Details / 模型说明
### Built-in Models / 内置模型
| Model Name               | English Full Name               | 中文名称               | Key Parameters Tuned                          | 调优参数                          |
|--------------------------|---------------------------------|-----------------------|-----------------------------------------------|-----------------------------------|
| SVM                      | Support Vector Machine Regression | 支持向量机回归         | `sigma` (10⁻³ to 10¹), `C` (10⁻¹ to 10³)       | `sigma`（10⁻³至10¹）、`C`（10⁻¹至10³） |
| RF                       | Random Forest Regression         | 随机森林回归           | `mtry` (1 to number of features)               | `mtry`（1至特征数量）              |
| GSL                      | Gaussian Process Regression      | 高斯过程回归           | `sigma` (10⁻³ to 10¹)                          | `sigma`（10⁻³至10¹）              |
| PLS                      | Partial Least Squares Regression | 偏最小二乘回归         | `ncomp` (1 to min(10, number of features))     | `ncomp`（1至10或特征数量的最小值） |
| NN                       | Neural Network                   | 神经网络               | `size` (5 to 25), `decay` (10⁻³ to 10⁰)        | `size`（5至25）、`decay`（10⁻³至10⁰） |

### Evaluation Metrics / 评估指标
- **R² (Coefficient of Determination)**: Measures how well the model fits the data (range: 0–1; higher = better).  
  **决定系数（R²）**：衡量模型拟合度（范围0–1，值越大越好）。  
- **RMSE (Root Mean Squared Error)**: Measures prediction error (unit same as CH; lower = better).  
  **均方根误差（RMSE）**：衡量预测误差（单位与CH一致，值越小越好）。  
- **MAE (Mean Absolute Error)**: Measures average absolute prediction error (lower = better).  
  **平均绝对误差（MAE）**：衡量平均绝对预测误差（值越小越好）。


## Results Interpretation / 结果说明
1. **Optimal Model Selection**: The script automatically selects the model with the **smallest test-set RMSE** as the optimal model for CH inversion.  
   **最优模型选择**：脚本自动将**测试集RMSE最小**的模型作为最优株高反演模型。  
2. **Result Files**:  
   - `拔节期-CH模型评估.csv`: Compare performance across models (e.g., RF often outperforms others for spectral data).  
     可对比各模型性能（如光谱数据中随机森林通常表现更优）。  
   - `拔节期-CH反演结果.csv`: Apply all models to the full dataset for batch prediction.  
     可将所有模型应用于全数据集进行批量预测。  
3. **Visualization**: The scatter plot includes:  
   - Training/test set points (blue/orange).  
   - 45° reference line (ideal "predicted = actual" scenario).  
   - Annotations of R² and RMSE for both datasets.  
   **可视化图表包含**：训练集（蓝色）/测试集（橙色）散点、45°参考线（理想“预测=实际”情况）、数据集R²和RMSE标注。


## License / 许可证
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.  
本项目基于**MIT许可证**开源 - 详见[LICENSE](LICENSE)文件。


## Contribution / 贡献说明
We welcome contributions to improve the project:  
欢迎通过以下方式贡献代码或优化功能：
- Submit **Issues** for bug reports or feature requests.  
  提交**Issue**反馈bug或建议新功能。  
- Create **Pull Requests** to fix bugs, optimize code, or add new models (e.g., XGBoost, LightGBM).  
  创建**Pull Request**修复bug、优化代码或添加新模型（如XGBoost、LightGBM）。


## Contact / 联系方式
If you have questions, feel free to contact:  
如有疑问，可通过以下方式联系：
- Email: [smy511@henau.edu.cn]

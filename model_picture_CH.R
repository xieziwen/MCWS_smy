# 清除工作区空间
rm(list = ls())

# 加载必要的库
library(pls)
library(car)
library(e1071)
library(caret)
library(glmnet)
library(gbm)
library(tidyverse)
library(kernlab)

# 设置工作目录
setwd("E:/1_wzy/corn/predict/data")

# 导入数据
res_band <- rio::import("数据汇总-拔节期.xlsx", sheet = 1)
processed_data <- res_band

# 数据划分
set.seed(247)
n <- nrow(processed_data)
indices <- sample(1:n)  

# 按 2:1 划分训练集和测试集
train_end <- floor(0.67 * n) 
test_end <- floor(0.33 * n)  

# 划分训练集和测试集
data_train <- processed_data[indices[1:train_end], ] 
data_test <- processed_data[indices[(train_end + 1):test_end], ]  

# 统一测试集的因子分类水平
factor_cols <- names(processed_data)[sapply(processed_data, is.factor)] 
for (col in factor_cols) {
  data_test[[col]] <- factor(data_test[[col]], levels = levels(data_train[[col]]))
}

# 数据预处理：PCA
# --------------------------------------------------
# 提取特征和目标变量
target <- "CH"
features_train <- data_train %>% select(-all_of(target))
features_test <- data_test %>% select(-all_of(target))

# 将因子特征转换为虚拟变量
if (length(factor_cols) > 0) {
  dummy <- dummyVars(~., data = features_train)
  features_train_dummy <- predict(dummy, features_train)
  features_test_dummy <- predict(dummy, features_test)
  
  # 对齐列名
  missing_cols <- setdiff(colnames(features_train_dummy), colnames(features_test_dummy))
  features_test_dummy <- cbind(
    features_test_dummy,
    matrix(0, 
           nrow = nrow(features_test_dummy),
           ncol = length(missing_cols),
           dimnames = list(NULL, missing_cols)
    )
  )
  features_test_dummy <- features_test_dummy[, colnames(features_train_dummy)]
} else {
  features_train_dummy <- as.matrix(features_train)
  features_test_dummy <- as.matrix(features_test)
}

# 执行PCA
pca_model <- prcomp(features_train_dummy, center = TRUE, scale. = TRUE)

# 确定保留的主成分数目（累计方差>=95%）
cum_var <- cumsum(pca_model$sdev^2)/sum(pca_model$sdev^2)
n_components <- which(cum_var >= 0.95)[1]
if (is.na(n_components)) n_components <- length(pca_model$sdev)

# 转换数据并设置列名
train_pca <- as.data.frame(predict(pca_model, features_train_dummy)[, 1:n_components])
test_pca <- as.data.frame(predict(pca_model, features_test_dummy)[, 1:n_components])
colnames(train_pca) <- paste0("PC", 1:n_components)
colnames(test_pca) <- paste0("PC", 1:n_components)

# 重建数据集
data_train <- data.frame(CH = data_train[[target]], train_pca)
data_test <- data.frame(CH = data_test[[target]], test_pca)
# --------------------------------------------------

# 定义目标变量
targets <- c("CH")  

# 定义存储列表
final_results <- list()
estimate_results <- data.frame(Actual_CH = data_test$CH)
models <- list()  # 新增模型存储列表

# 交叉验证
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# 模型训练和预测
for (target in targets) {
  formula <- as.formula(paste(target, "~ ."))
  
  # 获取当前特征数量
  n_features <- ncol(data_train) - 1
  
  # 支持向量机回归（SVM）
  tryCatch({
    svm_grid <- expand.grid(sigma = 10^seq(-3, 1, length = 5),
                            C = 10^seq(-1, 3, length = 5))
    models$svm <- train(formula, data = data_train, method = "svmRadial",
                        trControl = ctrl, tuneGrid = svm_grid)
    predict_svm_train <- predict(models$svm, data_train)
    predict_svm_test <- predict(models$svm, data_test)
  }, error = function(e) {
    message("SVM 模型训练失败: ", e$message)
  })
  
  # 随机森林回归（RF）
  tryCatch({
    rf_grid <- expand.grid(mtry = seq(1, n_features, by = 1))
    models$rf <- train(formula, data = data_train, method = "rf",
                       trControl = ctrl, tuneGrid = rf_grid, ntree = 500)
    predict_rf_train <- predict(models$rf, data_train)
    predict_rf_test <- predict(models$rf, data_test)
  }, error = function(e) {
    message("RF 模型训练失败: ", e$message)
  })
  
  # 高斯过程回归（GSL）
  tryCatch({
    gsl_grid <- expand.grid(sigma = 10^seq(-3, 1, length = 5))
    models$gsl <- train(formula, data = data_train, method = "gaussprRadial",
                        trControl = ctrl, tuneGrid = gsl_grid)
    predict_gsl_train <- predict(models$gsl, data_train)
    predict_gsl_test <- predict(models$gsl, data_test)
  }, error = function(e) {
    message("GSL 模型训练失败: ", e$message)
  })
  
  # 偏最小二乘回归（PLS）
  tryCatch({
    pls_grid <- expand.grid(ncomp = 1:min(10, n_features))
    models$pls <- train(formula, data = data_train, method = "pls",
                        trControl = ctrl, tuneGrid = pls_grid)
    predict_pls_train <- predict(models$pls, data_train)
    predict_pls_test <- predict(models$pls, data_test)
  }, error = function(e) {
    message("PLS 模型训练失败: ", e$message)
  })
  
  # 神经网络（NN）
  tryCatch({
    nn_grid <- expand.grid(size = seq(5, 25, by = 5),
                           decay = 10^seq(-3, 0, length = 5))
    models$nn <- train(formula, data = data_train, method = "nnet",
                       linout = TRUE, preProcess = c("center", "scale"),
                       maxit = 500, trControl = ctrl, tuneGrid = nn_grid)
    predict_nn_train <- predict(models$nn, data_train)
    predict_nn_test <- predict(models$nn, data_test)
  }, error = function(e) {
    message("NN 模型训练  败: ", e$message)
  })
  
  # 评估模型性能
  model_metrics <- data.frame(
    model = c("SVM", "RF", "GSL", "PLS", "NN"),
    R2_train = round(c(
      caret::R2(predict_svm_train, data_train[[target]]),
      caret::R2(predict_rf_train, data_train[[target]]),
      caret::R2(predict_gsl_train, data_train[[target]]),
      caret::R2(predict_pls_train, data_train[[target]]),
      caret::R2(predict_nn_train, data_train[[target]])
    ), 3),
    RMSE_train = round(c(
      caret::RMSE(predict_svm_train, data_train[[target]]),
      caret::RMSE(predict_rf_train, data_train[[target]]),
      caret::RMSE(predict_gsl_train, data_train[[target]]),
      caret::RMSE(predict_pls_train, data_train[[target]]),
      caret::RMSE(predict_nn_train, data_train[[target]])
    ), 3),
    MAE_train = round(c(
      caret::MAE(predict_svm_train, data_train[[target]]),
      caret::MAE(predict_rf_train, data_train[[target]]),
      caret::MAE(predict_gsl_train, data_train[[target]]),
      caret::MAE(predict_pls_train, data_train[[target]]),
      caret::MAE(predict_nn_train, data_train[[target]])
    ), 3),
    R2_test = round(c(
      caret::R2(predict_svm_test, data_test[[target]]),
      caret::R2(predict_rf_test, data_test[[target]]),
      caret::R2(predict_gsl_test, data_test[[target]]),
      caret::R2(predict_pls_test, data_test[[target]]),
      caret::R2(predict_nn_test, data_test[[target]])
    ), 3),
    RMSE_test = round(c(
      caret::RMSE(predict_svm_test, data_test[[target]]),
      caret::RMSE(predict_rf_test, data_test[[target]]),
      caret::RMSE(predict_gsl_test, data_test[[target]]),
      caret::RMSE(predict_pls_test, data_test[[target]]),
      caret::RMSE(predict_nn_test, data_test[[target]])
    ), 3),
    MAE_test = round(c(
      caret::MAE(predict_svm_test, data_test[[target]]),
      caret::MAE(predict_rf_test, data_test[[target]]),
      caret::MAE(predict_gsl_test, data_test[[target]]),
      caret::MAE(predict_pls_test, data_test[[target]]),
      caret::MAE(predict_nn_test, data_test[[target]])
    ), 3)
  )
  
  # 保存评估结果
  final_results[[target]] <- model_metrics
  
  # 选择最佳模型
  best_model <- model_metrics$model[which.min(model_metrics$RMSE_test)]
  best_predictions <- switch(
    best_model,
    SVM = predict_svm_test,
    RF = predict_rf_test,
    GSL = predict_gsl_test,
    PLS = predict_pls_test,
    NN = predict_nn_test
  )
  
  if (target == "CH") {
    estimate_results$Predicted_CH <- round(best_predictions, 3)
  }
}

# 全数据预测处理
# --------------------------------------------------
# 预处理全数据
full_features <- processed_data %>% select(-all_of(target))

if (length(factor_cols) > 0) {
  full_dummy <- predict(dummy, full_features)
  missing_cols <- setdiff(colnames(features_train_dummy), colnames(full_dummy))
  full_dummy <- cbind(
    full_dummy,
    matrix(0, 
           nrow = nrow(full_dummy),
           ncol = length(missing_cols),
           dimnames = list(NULL, missing_cols)
    )
  )[, colnames(features_train_dummy)]
} else {
  full_dummy <- as.matrix(full_features)
}

# PCA转换
full_pca <- as.data.frame(predict(pca_model, full_dummy)[, 1:n_components])
colnames(full_pca) <- paste0("PC", 1:n_components)
full_data <- data.frame(CH = processed_data[[target]], full_pca)

# 生成全数据预测
model_order <- c("svm", "rf", "gsl", "pls", "nn")
all_predictions <- data.frame(Actual_CH = full_data$CH)

for (model_name in model_order) {
  if (!is.null(models[[model_name]])) {
    pred <- predict(models[[model_name]], newdata = full_data)
    all_predictions[[paste0(toupper(model_name), "_Pred")]] <- round(pred, 3)
  }
}

# 输出预测结果
write.csv(all_predictions, 
          "E:/1_wzy/corn/predict/output/拔节期-CH反演结果.csv", 
          row.names = FALSE)
# --------------------------------------------------

# 输出评估结果
print(final_results)

# 合并评估结果
final_results_combined <- do.call(rbind, final_results)
write.csv(final_results_combined, 
          file = "E:/1_wzy/corn/predict/output/拔节期-CH模型评估.csv", 
          row.names = FALSE)

# 生成预测数据（确保模型已正确训练）
if(exists("models") && !is.null(models$rf)){
  predict_rf_train <- predict(models$rf, newdata = data_train)
  predict_rf_test <- predict(models$rf, newdata = data_test)
} else {
  stop("随机森林模型未正确训练，请先运行模型训练代码")
}

# 创建绘图数据集
plot_data <- rbind(
  data.frame(
    Actual = data_train$CH,
    Predicted = predict_rf_train,
    Dataset = "训练集"
  ),
  data.frame(
    Actual = data_test$CH,
    Predicted = predict_rf_test,
    Dataset = "测试集"
  )
)

# 计算所有指标
r2_train <- caret::R2(predict_rf_train, data_train$CH)
rmse_train <- caret::RMSE(predict_rf_train, data_train$CH)
rmse_test <- caret::RMSE(predict_rf_test, data_test$CH)
r2_test <- caret::R2(predict_rf_test, data_test$CH)

# 生成严格5等分的整数分界点
min_val <- floor(min(plot_data$Actual, plot_data$Predicted))
max_val <- ceiling(max(plot_data$Actual, plot_data$Predicted))
n_breaks <- 5  # 5个分界点对应4个间隔
interval <- ceiling((max_val - min_val) / (n_breaks - 1))  # 确保间隔为整数

# 生成等差序列（可能略微扩展坐标轴范围）
breaks <- seq(
  from = min_val,
  by = interval,
  length.out = n_breaks
)

# 可视化实现
ggplot(plot_data, aes(x = Actual, y = Predicted, color = Dataset)) +
  geom_point(alpha = 0.7, size = 2.5, shape = 16) +
  geom_abline(
    slope = 1, intercept = 0, 
    color = "black", linetype = "solid", linewidth = 0.8
  ) +
  scale_x_continuous(
    breaks = breaks,
    limits = range(breaks),
    name = "Measured CH (cm)",
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    breaks = breaks,
    limits = range(breaks),
    name = "Estimated CH (cm)",
    expand = c(0.02, 0.02)
  ) +
  theme_bw(base_size = 12) +
  labs(
    x = NULL,  # 禁用默认x标签
    y = NULL,   # 禁用默认y标签
    title = "(a)"
  ) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.text = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, face = "bold"),  # 加粗坐标轴名称
    plot.title = element_text(hjust = 0.5, size = 12),    # 居中标题
    legend.position = "none"
  ) +
  scale_color_manual(values = c("训练集" = "#1f77b4", "测试集" = "#ff7f0e")) +
  
 
  
  # 训练集指标
  annotate("text", x = min_val + 0.5, y = max_val - 6,
           label = sprintf("● Training set  R² = %.2f, RMSE = %.2f (cm)", 
                           r2_train, rmse_train),
           hjust = 0, vjust = 0, color = "#1f77b4",
           size = 4, lineheight = 0.8) +
  
  # 测试集指标
  annotate("text", x = min_val + 0.5, y = max_val - 12,
           label = sprintf("● Testing set  R² = %.2f, RMSE = %.2f (cm)", 
                           r2_test, rmse_test),
           hjust = 0, vjust = 0, color = "#ff7f0e",
           size = 4, lineheight = 0.8)



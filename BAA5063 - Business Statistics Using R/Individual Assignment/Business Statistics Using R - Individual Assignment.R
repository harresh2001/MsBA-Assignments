# Load necessary libraries
library(readr)  # For reading CSV files
library(dplyr)  # For data manipulation
library(ggplot2)  # For visualizations
library(caret)  # For model training
library(tidyr)  # For data reshaping
library(corrplot)
library(gridExtra)
library(car)
library(pROC)

# Load the data
df <- read_csv("employee_churn_data.csv")

# Initial exploration
head(df)
str(df)
summary(df)

# Check for missing values
colSums(is.na(df))
# Check for duplicates and handle them
if (any(duplicated(df))) {
  df <- df[!duplicated(df), ]
  print("Duplicates have been removed.")
} else {
  print("No duplicate rows found.")
}

# Create histograms for numerical variables
ggplot(df, aes(x = review)) + 
  geom_histogram(bins = 20, fill = "skyblue", color = "black") + 
  labs(title = "Distribution of 'review'")

ggplot(df, aes(x = satisfaction)) + 
  geom_histogram(bins = 20, fill = "lightcoral", color = "black") + 
  labs(title = "Distribution of 'satisfaction'")

ggplot(df, aes(x = avg_hrs_month)) + 
  geom_histogram(bins = 20, fill = "plum", color = "black") + 
  labs(title = "Distribution of 'avg_hrs_month'")

# Visualizing the distribution of the target variable 'left'
ggplot(df, aes(x = factor(left))) +
  geom_bar() +
  labs(x = "Employee Turnover", y = "Count", title = "Distribution of Employee Turnover")

# Visualizing relationships between categorical variables and the target 'event'
ggplot(df, aes(x = factor(department), fill = factor(left))) +
  geom_bar(position = "dodge") +
  labs(title = "Department vs. Employee Turnover")

ggplot(df, aes(x = factor(promoted), fill = factor(left))) +
  geom_bar(position = "dodge") +
  labs(title = "Promotion vs. Employee Turnover")

ggplot(df, aes(x = factor(projects), fill = factor(left))) +
  geom_bar(position = "dodge") +
  labs(title = "Number of Projects vs. Employee Turnover")

ggplot(df, aes(x = factor(salary), fill = factor(left))) +
  geom_bar(position = "dodge") +
  labs(title = "Salary vs. Employee Turnover")

ggplot(df, aes(x = factor(tenure), fill = factor(left))) +
  geom_bar(position = "dodge") +
  labs(title = "Tenure vs. Employee Turnover")

ggplot(df, aes(x = factor(bonus), fill = factor(left))) +
  geom_bar(position = "dodge") +
  labs(title = "Bonus vs. Employee Turnover")

# Convert 'left' to numeric (0 = 'no', 1 = 'yes')
df$left <- ifelse(df$left == "yes", 1, 0)

# Perform one-hot encoding for the 'department' and 'salary' column
department_encoded <- model.matrix(~ department - 1, data = df)
salary_encoded <- model.matrix(~ salary - 1, data = df)

# Combine the encoded columns with the original dataframe
df <- cbind(df, department_encoded)
df <- cbind(df, salary_encoded)
df$department <- NULL
df$salary <- NULL

# Calculate correlations only for numeric variables
cor_matrix <- cor(df[, c("review", "projects", "tenure", "satisfaction", "avg_hrs_month", "left")])

# View the correlation matrix
print(cor_matrix)


# Create a heatmap of the correlation matrix
corrplot(cor_matrix, method = "color", 
         col = colorRampPalette(c("red", "white", "blue"))(200),
         type = "full",    # Display the upper triangle only
         tl.col = "black",  # Text label color
         tl.srt = 45,       # Rotate text labels
         addCoef.col = "black",  # Add correlation coefficients
         number.cex = 0.8)  # Adjust the size of numbers

# Plot boxplots for numeric variables
df %>%
  dplyr::select(review) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal()

df %>%
  dplyr::select(satisfaction) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal()

df %>%
  dplyr::select(avg_hrs_month) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal()

# Function to detect outliers using IQR method and return count
detect_outliers_IQR <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR_value <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  outliers <- which(x < lower_bound | x > upper_bound)
  outlier_count <- length(outliers)  # Count the number of outliers
  return(list(outliers = outliers, count = outlier_count))
}

# Detect outliers for each column
outliers_review <- detect_outliers_IQR(df$review)

# View indices of outliers and their counts
outliers_review$outliers
outliers_review$count

# Remove rows with any outliers in the "review" column
df_cleaned <- df[-outliers_review$outliers, ]

# Plot boxplots for numeric variables
df_cleaned %>%
  dplyr::select(review) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal()

# List of columns to convert to factors
columns_to_convert <- c("promoted", "bonus", "left", "departmentadmin", "departmentengineering", "departmentfinance", 
                        "departmentIT", "departmentlogistics", "departmentmarketing", 
                        "departmentoperations", "departmentretail", "departmentsales", 
                        "departmentsupport", "salaryhigh", "salarylow", "salarymedium")

# Convert the specified columns to factors
df_cleaned[columns_to_convert] <- lapply(df_cleaned[columns_to_convert], as.factor)

# Remove the 'departmentsupport' and 'salarymedium' column from the dataframe
df_cleaned <- df_cleaned[, !colnames(df_cleaned) %in% "departmentsupport"]
df_cleaned <- df_cleaned[, !colnames(df_cleaned) %in% "salarymedium"]

# Set a seed for reproducibility
set.seed(123)

# Split the data into training and testing sets (80% training, 20% testing)
trainIndex <- createDataPartition(df_cleaned$left, p = 0.8, list = FALSE)
train_data <- df_cleaned[trainIndex, ]
test_data <- df_cleaned[-trainIndex, ]

# Fit logistic regression model on training data
model1 <- glm(left ~ review + projects + tenure + satisfaction + bonus + avg_hrs_month + 
                departmentadmin + departmentengineering + departmentfinance + departmentIT + 
                departmentlogistics + departmentmarketing + departmentoperations + departmentretail + 
                departmentsales + salaryhigh + salarylow, 
              data = train_data, 
              family = binomial)

# Summarize the model results
summary(model1)

# VIF for multicollinearity
vif(model1)

# Fit logistic regression model on training data (removed 'tenure')
model2 <- glm(left ~ review + projects + satisfaction + bonus + avg_hrs_month + 
                departmentadmin + departmentengineering + departmentfinance + departmentIT + 
                departmentlogistics + departmentmarketing + departmentoperations + departmentretail + 
                departmentsales + salaryhigh + salarylow, 
              data = train_data, 
              family = binomial)

# Summarize the model results
summary(model2)

# VIF for multicollinearity
vif(model2)

# Predict probabilities on test data
test_data$predicted_prob <- predict(model2, newdata = test_data, type = "response")

# Convert probabilities to binary class predictions
test_data$predicted_class <- ifelse(test_data$predicted_prob > 0.5, 1, 0)

# View the confusion matrix for evaluation
confusion_matrix <- table(Predicted = test_data$predicted_class, Actual = test_data$left)
print(confusion_matrix)

# Extract values from the confusion matrix
true_positive <- confusion_matrix[2, 2] # Predicted 1, Actual 1
true_negative <- confusion_matrix[1, 1] # Predicted 0, Actual 0
false_positive <- confusion_matrix[2, 1] # Predicted 1, Actual 0
false_negative <- confusion_matrix[1, 2] # Predicted 0, Actual 1

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy: ", round(accuracy, 2)))

# Calculate precision
precision <- true_positive / (true_positive + false_positive)
print(paste("Precision: ", round(precision, 2)))

# Calculate recall (same as sensitivity)
recall <- true_positive / (true_positive + false_negative)
print(paste("Recall (Sensitivity): ", round(recall, 2)))

# Calculate specificity
specificity <- true_negative / (true_negative + false_positive)
print(paste("Specificity: ", round(specificity, 2)))

# Calculate F1-score
f1_score <- 2 * ((precision * recall) / (precision + recall))
print(paste("F1-Score: ", round(f1_score, 2)))

# Plot ROC curve for model performance
roc_curve <- roc(test_data$left, test_data$predicted_prob)
plot(roc_curve, main = "ROC Curve", col = "blue")

# Calculate the AUC
auc_value <- auc(roc_curve)
print(paste("AUC: ", round(auc_value, 2)))

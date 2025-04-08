# load library
library(readr)     # For reading csv file
library(plm)       # For panel data models
library(lmtest)    # For panel data models
library(gplots)    # For panel data plots
library(tidyverse) # For data management
library(foreign)

# read data
df <- read_csv("C:/Users/yvonn/OneDrive/Desktop/SUNWAY/202409-S4/Business Intelligence/Assignment/SIandHappinessV2.csv")

# Rename all columns to remove spaces and special characters
colnames(df) <- gsub(" ", "_", colnames(df))  # Replace spaces with underscores
colnames(df) <- gsub("[^[:alnum:]_]", "", colnames(df))  # Remove special characters
# Check the updated column names
colnames(df)

# Replace double underscores with a single underscore
colnames(df) <- gsub("__", "_", colnames(df))
# ensure no leading or trailing underscores
colnames(df) <- gsub("^_|_$", "", colnames(df))  # Remove leading or trailing underscores
# Check the updated column names
print(colnames(df))

# prepare the data for panel analysis
# convert to panel data frame ('country_name' and 'year' as identifiers)
df_panel <- pdata.frame(df, index = c("Country_name", "year"))


## Unobserved heterogeneity
plotmeans(Life_Ladder ~ Country_name, main="Heterogeneity across countries", data = df_panel)
plotmeans(Life_Ladder ~ year, main="Heterogeneity across time", data = df_panel)

# Descriptive Stats
summary(df_panel)

# there is NaN produced,Check for missing values in the dataset
colSums(is.na(df_panel))

# Create a new dataframe df_cleaned and impute missing values
df_cleaned <- df_panel

# Impute missing values using the mean for each numeric column in df_cleaned
df_cleaned$Nutrition_Basic_Medical_Care[is.na(df_cleaned$Nutrition_Basic_Medical_Care)] <- mean(df_cleaned$Nutrition_Basic_Medical_Care, na.rm = TRUE)
df_cleaned$Water_Sanitation[is.na(df_cleaned$Water_Sanitation)] <- mean(df_cleaned$Water_Sanitation, na.rm = TRUE)
df_cleaned$Shelter[is.na(df_cleaned$Shelter)] <- mean(df_cleaned$Shelter, na.rm = TRUE)
df_cleaned$Personal_Safety[is.na(df_cleaned$Personal_Safety)] <- mean(df_cleaned$Personal_Safety, na.rm = TRUE)
df_cleaned$Access_to_Basic_Knowledge[is.na(df_cleaned$Access_to_Basic_Knowledge)] <- mean(df_cleaned$Access_to_Basic_Knowledge, na.rm = TRUE)
df_cleaned$Access_to_Information_Communications[is.na(df_cleaned$Access_to_Information_Communications)] <- mean(df_cleaned$Access_to_Information_Communications, na.rm = TRUE)
df_cleaned$Health_Wellness[is.na(df_cleaned$Health_Wellness)] <- mean(df_cleaned$Health_Wellness, na.rm = TRUE)
df_cleaned$Environmental_Quality[is.na(df_cleaned$Environmental_Quality)] <- mean(df_cleaned$Environmental_Quality, na.rm = TRUE)
df_cleaned$Personal_Rights[is.na(df_cleaned$Personal_Rights)] <- mean(df_cleaned$Personal_Rights, na.rm = TRUE)
df_cleaned$Personal_Freedom_Choice[is.na(df_cleaned$Personal_Freedom_Choice)] <- mean(df_cleaned$Personal_Freedom_Choice, na.rm = TRUE)
df_cleaned$Inclusiveness[is.na(df_cleaned$Inclusiveness)] <- mean(df_cleaned$Inclusiveness, na.rm = TRUE)
df_cleaned$Access_to_Advanced_Education[is.na(df_cleaned$Access_to_Advanced_Education)] <- mean(df_cleaned$Access_to_Advanced_Education, na.rm = TRUE)

# Verify the missing values after imputation
colSums(is.na(df_cleaned))

# Analysis
# Pooled OLS Model
pooled_ols <- plm(Life_Ladder ~ Nutrition_Basic_Medical_Care + Water_Sanitation 
                  + Shelter + Personal_Safety + Access_to_Basic_Knowledge 
                  + Access_to_Information_Communications + Health_Wellness 
                  + Environmental_Quality + Personal_Rights 
                  + Personal_Freedom_Choice + Inclusiveness 
                  + Access_to_Advanced_Education, 
                  data = df_cleaned, model = "pooling")

# Summarize the results
summary(pooled_ols)

# Random Effects Model
random_effects <- plm(Life_Ladder ~ Nutrition_Basic_Medical_Care + Water_Sanitation 
                      + Shelter + Personal_Safety + Access_to_Basic_Knowledge 
                      + Access_to_Information_Communications + Health_Wellness 
                      + Environmental_Quality + Personal_Rights 
                      + Personal_Freedom_Choice + Inclusiveness 
                      + Access_to_Advanced_Education,
                      data = df_cleaned, model = "random")

summary(random_effects)

# (to test pooled ols or random) Breusch-Pagan LM Test
bp_test <- bptest(pooled_ols)
print(bp_test)

# Fixed Effects Model
fixed_effects <- plm(Life_Ladder ~ Nutrition_Basic_Medical_Care + Water_Sanitation 
                     + Shelter + Personal_Safety + Access_to_Basic_Knowledge 
                     + Access_to_Information_Communications + Health_Wellness 
                     + Environmental_Quality + Personal_Rights 
                     + Personal_Freedom_Choice + Inclusiveness 
                     + Access_to_Advanced_Education,
                     data = df_cleaned, model = "within")

summary(fixed_effects)

## statistics from FE model
# Display the fixed effects (constants for each country)
fixef(fixed_effects)

# Hausman Test If p value is < 0.05 then use fixed effects
phtest(fixed_effects, random_effects)

# Testing for fixed effects, null: OLS better than fixed
pFtest(fixed_effects, pooled_ols)

# since Hausman test reveal that should use ficed effects, so we focus on fixed effect here
fixed_effects <- plm(Life_Ladder ~ Nutrition_Basic_Medical_Care + Water_Sanitation 
                     + Shelter + Personal_Safety + Access_to_Basic_Knowledge 
                     + Access_to_Information_Communications + Health_Wellness 
                     + Environmental_Quality + Personal_Rights 
                     + Personal_Freedom_Choice + Inclusiveness 
                     + Access_to_Advanced_Education,
                     data = df_cleaned, model = "within")
summary(fixed_effects)




# robustness check (Time-Fixed Effects: To control for time-specific shocks that might affect Life Ladder scores, include time-fixed effects)
fixed_effects_time <- plm(Life_Ladder ~ Nutrition_Basic_Medical_Care + Water_Sanitation 
                          + Shelter + Personal_Safety + Access_to_Basic_Knowledge 
                          + Access_to_Information_Communications + Health_Wellness 
                          + Environmental_Quality + Personal_Rights 
                          + Personal_Freedom_Choice + Inclusiveness 
                          + Access_to_Advanced_Education + factor(year),
                          data = df_cleaned, model = "within")
summary(fixed_effects_time)

# Testing time-fixed effects. The null is that no time-fixed effects needed
pFtest(fixed_effects_time, fixed_effects)

plmtest(fixed_effects, c("time"), type=("bp"))

#Testing for cross-sectional dependence/contemporaneous correlation:
#using Breusch-Pagan LM test of independence and Pasaran CD test
pcdtest(fixed_effects_time, test = c("lm"))

pcdtest(fixed_effects_time, test = c("cd"))


# Compute Driscoll and Kraay standard errors
dk_se <- vcovDC(fixed_effects_time, type = "HC3")

# Display summary with adjusted standard errors
summary(fixed_effects_time, vcov = dk_se)


#Visualizeeeeee
library(dplyr)

# Assuming 'fixed_effects_time'
dk_summary <- summary(fixed_effects_time, vcov = dk_se)

# Extract coefficients, standard errors, and p-values
results <- data.frame(
  Variable = rownames(dk_summary$coefficients),
  Estimate = dk_summary$coefficients[, "Estimate"],
  Std_Error = dk_summary$coefficients[, "Std. Error"],
  P_Value = dk_summary$coefficients[, "Pr(>|t|)"]
)

# Calculate confidence intervals
results <- results %>%
  mutate(
    Lower_CI = Estimate - 1.96 * Std_Error,
    Upper_CI = Estimate + 1.96 * Std_Error,
    Significance = case_when(
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ ""
    )
  )



library(gridExtra)

# Create the individual plots
p1 <- ggplot(results, aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Lower_CI, ymax = Upper_CI), width = 0.2) +
  coord_flip() +
  labs(title = "Coefficient Plot", x = "Variables", y = "Coefficient Estimate") +
  theme_minimal()

p2 <- ggplot(results, aes(x = reorder(Variable, Estimate), y = Estimate, fill = Significance)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Bar Plot of Coefficient Estimates", x = "Variables", y = "Coefficient Estimate") +
  theme_minimal()

# Combine the plots
grid.arrange(p1, p2, ncol = 2)









##Model training
library(caret)

# Set seed for reproducibility
set.seed(123)

# Split the data
trainIndex <- createDataPartition(df_cleaned$Life_Ladder, p = .7, 
                                  list = FALSE, 
                                  times = 1)
df_train <- df_cleaned[trainIndex, ]
df_test <- df_cleaned[-trainIndex, ]

#Train using fixed time effect model
model <- fixed_effects_time
summary(model)

#predict using test data
pred <- predict(model,df_test)

# Check for NA values in predictions
sum(is.na(pred))

# check performance
# check RMSE
RMSE(pred, df_test$Life_Ladder)

# Custom MAPE function (Mean Absolute Percentage Error)
mape <- function(actual, predicted) {
  mean(abs((actual - predicted) / actual)) * 100
}

# Calculate MAPE using the custom function
mape_value <- mape(df_test$Life_Ladder, pred)
print(mape_value)


# Create a data frame for plotting
plot_df <- data.frame(
  Index = 1:nrow(df_test),  # Use the number of rows in the test set
  Actual = df_test$Life_Ladder,  # Actual values from the test set
  Predicted = pred  # Predicted values from the model
)

# Create a line plot
ggplot(plot_df, aes(x = Index)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 1) +
  labs(title = "Actual vs. Predicted Values",
       x = "Index",
       y = "Values") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "orange")) +
  theme_minimal()
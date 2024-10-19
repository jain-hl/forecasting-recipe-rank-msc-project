import pandas as pd
from scipy import stats

# Load the CSV files with the specified paths
linear_reg_csv = 'c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/LinearReg_shortterm_MAE.csv'
lightgbm_csv = 'c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/LightGBM_shortterm_full.csv'
realtime_feature_csv = 'c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/RealTimeFeature_MAE.csv'
piecewise_csv = 'c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/Piecewise_shortterm_full.csv'

# Load the datasets into DataFrames
linear_reg_df = pd.read_csv(linear_reg_csv)
lightgbm_df = pd.read_csv(lightgbm_csv)
realtime_feature_df = pd.read_csv(realtime_feature_csv)
piecewise_df = pd.read_csv(piecewise_csv)

# Set significance level (alpha)
alpha = 0.05

# Function to perform two-tailed t-test for each days_before (17 to -6) with summary statistics
def perform_t_test(model_a_df, model_b_df, model_a_name, model_b_name, days_before_value):
    # Filter the data for the specified days_before value
    model_a_filtered = model_a_df[model_a_df['days_before'] == days_before_value]
    model_b_filtered = model_b_df[model_b_df['days_before'] == days_before_value]

    # Ensure the max_week values are aligned between the two models
    merged_df = pd.merge(model_a_filtered[['max_week', 'MAE']], 
                         model_b_filtered[['max_week', 'MAE']], 
                         on='max_week', 
                         suffixes=('_a', '_b'))

    # Calculate summary statistics
    a_mean = merged_df['MAE_a'].mean()
    b_mean = merged_df['MAE_b'].mean()
    a_std = merged_df['MAE_a'].std()
    b_std = merged_df['MAE_b'].std()

    # Calculate the 95% confidence intervals
    n = merged_df.shape[0]
    a_se = a_std / (n ** 0.5)  # Standard Error
    b_se = b_std / (n ** 0.5)
    
    a_ci = stats.t.interval(0.95, n-1, loc=a_mean, scale=a_se)
    b_ci = stats.t.interval(0.95, n-1, loc=b_mean, scale=b_se)

    # Round the confidence intervals to 6 decimal places
    a_ci_rounded = (round(a_ci[0], 6), round(a_ci[1], 6))
    b_ci_rounded = (round(b_ci[0], 6), round(b_ci[1], 6))

    # Perform a paired t-test (two-tailed) comparing the MAE values
    t_stat, p_value_two_tailed = stats.ttest_rel(merged_df['MAE_a'], merged_df['MAE_b'])

    # Calculate the critical values for the two-tailed test
    critical_value_upper = stats.t.ppf(1 - alpha / 2, n - 1)  # Upper critical value
    critical_value_lower = stats.t.ppf(alpha / 2, n - 1)      # Lower critical value

    # Print the summary statistics and t-test results
    print(f"Results Summary for days_before = {days_before_value} ({model_a_name} vs {model_b_name}):")
    print(f"{model_a_name}: Mean = {a_mean:.6f}, Std Dev = {a_std:.6f}, 95% CI = {a_ci_rounded}")
    print(f"{model_b_name}: Mean = {b_mean:.6f}, Std Dev = {b_std:.6f}, 95% CI = {b_ci_rounded}")
    print(f"T-statistic: {t_stat:.4f}, Two-tailed P-value: {p_value_two_tailed:.6f}")

    # Determine significance for two-tailed test
    if abs(t_stat) > critical_value_upper:  # Check against the upper critical value for rejection
        print(f"Result: Reject the null hypothesis at alpha = {alpha}")
        if t_stat < 0:
            print(f"Conclusion: {model_a_name} performs significantly better than {model_b_name}.")
        else:
            print(f"Conclusion: {model_b_name} performs significantly better than {model_a_name}.")
    else:
        print(f"Result: Fail to reject the null hypothesis at alpha = {alpha}")
        print(f"Conclusion: {model_a_name} performs similarly to {model_b_name}.")
    print('-' * 50)

# Perform two-tailed t-tests for each days_before from 17 to -6 for LightGBM vs. Linear Regression
for days_before in range(17, -7, -1):
    perform_t_test(lightgbm_df, linear_reg_df, 'LightGBM', 'Linear Regression', days_before)

# Perform two-tailed t-tests for each days_before from 17 to -6 for Piecewise vs. Real-Time Feature
for days_before in range(17, -7, -1):
    perform_t_test(piecewise_df, realtime_feature_df, 'Piecewise', 'Real-Time Feature', days_before)

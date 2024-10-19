import pandas as pd
from scipy import stats

# Load the CSV files with the specified paths
xgb_df = pd.read_csv('c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/XGBoost_longterm_full.csv')
cbr_df = pd.read_csv('c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/CBRanker_longterm_full.csv')
lgbm_df = pd.read_csv('c:/Users/jainh/OneDrive/Documents/UCL DSML/MSc Project/code/data/LGBMRanker_longterm_full.csv')

# Set significance level (alpha)
alpha = 0.05

# Function to perform two-tailed t-test for each weeks_before (8, 3, 1) with summary statistics
def perform_t_test(model_a_df, model_b_df, model_a_name, model_b_name, weeks_before_value):
    # Filter the data for the specified weeks_before value
    model_a_filtered = model_a_df[model_a_df['weeks_before'] == weeks_before_value]
    model_b_filtered = model_b_df[model_b_df['weeks_before'] == weeks_before_value]

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

    # Round the confidence intervals to 4 decimal places
    a_ci_rounded = (round(a_ci[0], 4), round(a_ci[1], 4))
    b_ci_rounded = (round(b_ci[0], 4), round(b_ci[1], 4))

    # Perform a paired t-test (two-tailed) comparing the MAE values
    t_stat, p_value_two_tailed = stats.ttest_rel(merged_df['MAE_a'], merged_df['MAE_b'])

    # Calculate the critical values for the two-tailed test
    critical_value_upper = stats.t.ppf(1 - alpha / 2, n - 1)  # Upper critical value
    critical_value_lower = stats.t.ppf(alpha / 2, n - 1)      # Lower critical value

    # Print the summary statistics and t-test results
    print(f"Results Summary for weeks_before = {weeks_before_value} ({model_a_name} vs {model_b_name}):")
    print(f"{model_a_name}: Mean = {a_mean:.4f}, Std Dev = {a_std:.4f}, 95% CI = {a_ci_rounded}")
    print(f"{model_b_name}: Mean = {b_mean:.4f}, Std Dev = {b_std:.4f}, 95% CI = {b_ci_rounded}")
    print(f"T-statistic: {t_stat:.4f}, Two-tailed P-value: {p_value_two_tailed:.6f}")
    print(f"Critical Values for Two-tailed Test: [{critical_value_lower:.4f}, {critical_value_upper:.4f}]")

    # Determine significance for two-tailed test
    if abs(t_stat) > critical_value_upper:  # Check against the upper critical value for rejection
        print(f"Result: Reject the null hypothesis at alpha = {alpha}")
        if t_stat < 0:  # Change this to check if t_stat is negative
            print(f"Conclusion: {model_a_name} performs significantly better than {model_b_name}.")
        else:
            print(f"Conclusion: {model_b_name} performs significantly better than {model_a_name}.")
    else:
        print(f"Result: Fail to reject the null hypothesis at alpha = {alpha}")
    print('-' * 50)

# Perform two-tailed t-tests for weeks_before 8, 3, and 1 for XGBoost vs. CBRanker
for weeks_before in [8, 3, 1]:
    perform_t_test(xgb_df, cbr_df, 'XGBoost', 'CBRanker', weeks_before)

# Perform two-tailed t-tests for weeks_before 8, 3, and 1 for XGBoost vs. LightGBM Ranker
for weeks_before in [8, 3, 1]:
    perform_t_test(xgb_df, lgbm_df, 'XGBoost', 'LightGBM Ranker', weeks_before)

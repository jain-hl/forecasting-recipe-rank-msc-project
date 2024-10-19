# Forecasting Recipe Rank: Real-Time Prediction Framework

## Abstract

Machine Learning applications are becoming increasingly prominent within retail businesses, for forecasting product sales or service adoption. The meal-kit delivery subdivision within the food industry presents the challenge of predicting the popularity of menu items in advance, for inventory management optimisation and food waste reduction. The UK hosts many subscription-based meal-kit startups, with one prominent player being `Gousto', whose business model has potential to benefit from accurate forecasting of recipe uptake within their weekly menus.

This study explores various statistical, supervised learning and learning-to-rank algorithms to predict the ranks of Gousto recipe uptakes in a given weekly menu, alongside the custom time-series experimental setup required to create effective long-term and short-term evaluation frameworks. The results indicate XGBoost as the superior long-term model with a 10.09\% MAE 3 weeks prior to delivery. A piecewise LightGBM and Linear model attains a 2.49\% MAE with a 7 day lag, dropping to 1.40\% MAE 3 days in advance, constituting a short-term forecast which uses real-time customer selection data.

This study concludes that learning-to-rank algorithms surprisingly underperform in this ranking task, including a state-of-the-art LambdaMART implementation via LightGBM Ranker, with suggested explanations and future optimisations required to better leverage the functionalities of ranking models.

View the full report [here]

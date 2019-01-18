# wine-quality

The purpose of this repository is to examine what makes a wine good quality. Here we have two different datasets, *winequality-red.csv* and *winequality-white.csv* (obtained from the UCI Machine Learning Repository), where I will performing two different approaches to understanding the target variable, quality.

## [Red Wine Quality](red-wine/md-files/1.red-wine-exploratory-data-analysis.md)
In *red-wine-quality.ipynb*, my goal is to solve the problem: what features make a red wine stand out and how to judge wine quality without an expert always rating it. This is important because if you are a store selling wine, you want to be able to recommend your customers good quality wine without having to hire an expert to recognize those wines for you. For this task, I treat the dataset as a classification problem, where I use models such as Random Forest, Logistic Regression, and Support Vector machines to maximize on the metric: precision. 

## [White Wine Quality](white-wine/md-files/white-wine-quality.md)
On the other hand, in *white-wine-quality.ipynb*, my goal is to solve problem: what exact features make a white wine excellent and how does each feature impact the quality. This is useful for wine producers to recognize what aspect in their wines is praised by the experts, and to capitialize on that by making as much 'good' quality wine as possible. For this task, I utilize regression models such as Linear Regression and Logistic Regression to understand the impact each feature has on white wine. 

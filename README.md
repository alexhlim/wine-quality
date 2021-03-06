# Wine Quality
What exactly determines a wine's quality? This repository's main goals is to examine this question by looking deeper into the different factors contributing to a bottle of wine's overall quality level. Presented within this project are two different datasets: *winequality-red.csv* and *winequality-white.csv* (obtained from the UCI Machine Learning Repository). These datasets will be used to assist in the performance of two different approaches, classification and regression, to help us better comprehend the components of a high-quality wine.

## [Red Wine Quality](red-wine/md-files/1.red-wine-exploratory-data-analysis.md)
We are presented with one main question: What exactly determines the quality of a red wine and how can we use that information to rate it? 

Almost 32 billion bottles of wine are sold worldwide every year, yet the market's standards for what constitutes as a "good quality wine" differs from region to region and from drinker to drinker. I want to take this dataset and create a more standardized quality test for the every day drinker. How can I help an average person understand wine by looking at a few specific factors that will lead him/her to make more educated decisions when choosing which bottle to purchase?

Within the dataset, we have exactly 1,599 different red wines. Each wine has their own rating on various features such as alcohol content and pH level out of a score of ten, rated by industry experts. For this test, I am treating the dataset as a classification problem; I will use models such as Random Forest, Logistic Regression and Support Vector machinese to maximize on the metric, precision. 

## [White Wine Quality](white-wine/md-files/white-wine-quality.md)
How can we assist winemakers create more profitable wine in the ever-growing wine industry? 

The alcohol industry, especially wine, is one that people have the most confident in and for one simple reason: people simply are not going to stop drinking! The cost of wine-production is steadily increasing in the past years and the stress of creating the perfect barrel of wine has continuously increased over climate changes that affect winemakers' grape varietals. It is crucial for wine producers to recognize how experts are rating their wines since every day consumers rely on these ratings to make their purchasing choices. By focusing on these key features, wine producers can more efficiently produce the highest quality white wine possible.   

In this project, the goal is identify the key factors contributing to the production of a "good" white wine. This means identifying specific features that directly impact the wines' quality and by what magnitude. We have a dataset of exactly 4898 different white wines and their corresponding features that were rated by industry experts out of a score of ten. Within this test, I will be exploring the dataset and utilizing regression models such as Linear Regression and Logistic Regression to better understand the imapct of each feature of a wine on the wine's overall quality.

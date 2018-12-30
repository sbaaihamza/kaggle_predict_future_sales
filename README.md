# kaggle_predict_future_sales

In this competition i worked with a challenging time-series dataset consisting of daily sales data, provided by - 1C Company. The goal was to predict total sales for every product and store in the next month. 
By solving this competition i was able to apply and enhance my data science skills.
Main methods I used for this competition is LightGBM that provide the Leaderboard score: 0.90807(RMSE).

The most important features are:
* standard lag features (target value for X-months ago, target mean value for shop/item_category/items) (X=1,2,3,6,12)
* how much time this item is on the market (or is it brand new item that appear for the very first time)
* mean target values for new items (that appear for the very first time) for shop/item_category
* text features  generated from shop_names
* grouping items into small categories 

Tools I used in this competition are: numpy, pandas, sklearn, XGBoost GPU, LightGBM
All models are tuned on a windows10 with Intel i5 8thgen processor, 8GB RAM. Tuning models took about 12 to 16 hours, and training on the whole dataset took less than 15 minutes


# I. Exploratory Data Analysis
**Information can be found in  2 EDA notebooks**

* Take a quick look: -Import each dataset and make a Quick exploration for each one 
- Detect possible  outliers and missing values using (describe() and  hist() )
* Deeper investigation through some ambiguous or potential variables (Univariate Exploration): builds a solide foundation about the distribution and the structure of each dataset 
* Data cleaning:
   - item_cnt_day variable is heavily skewed, most of the values are arround 0 and 5.0 , this range contain 99.9% of the data range. but before droping them we need to check if those are a real outliers or special cases ( should check other items with item-id for those outlier items ).
-my investigation lead me to drop some rows that are clearly outliers from their influence on the distribution of that specific item
-i will do the same for item_price variable as item_cnt_day
-for the shop dataset i found out that several shops are duplicates of each other (according to its name), so i fix the training and testings set
 * Discover and Visualize to gain insights: 
-using summary statistics its hepls but not that much cause we're dealing with time-series datasets
-so what realy helps is Exploratory Analysis(Bivariate Exploration specifically) using the date variable on the x-axis  and other variable on the y-axis

# II. Feature Engineering
**Information can be found in feature_eng  notebook**

-Category_item and shop_items shows strong decreasing trend and yearly seasonal pattern, therefore, should incorporate lag 12 features. Autocorrelation plot shows the previous 6 months often have positive correlation, therefore include lag 1 to 6 features.


* Prepare the data for ML algo

# III. Cross validations
**Information can be found in function define within feature_eng  notebook called get_cv_idxs()**

# IV. Training methods:
**Information can be found in tuning_lgb and tuning_xgb notebook**

# V. Ensembling
**Information can be found in ensemble notebook**


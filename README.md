# kaggle_predict_future_sales

In this competition i worked with a challenging time-series dataset consisting of daily sales data, provided by - 1C Company. The goal was to predict total sales for every product and store in the next month. 
By solving this competition i was able to apply and enhance my data science skills.
Main methods I used for this competition  is that provides LightGBM the  Leaderboard score: 0.90807(RMSE).

The most important features are:
* standard lag features (target value for X-months ago, target mean value for shop/item_category/items) (X=1,2,3,6,12)
* how much time this item is on the market (or is it brand new item that appear for the very first time)
* mean target values for new items (that appear for the very first time) for shop/item_category
* text features  generated from shop_names
* grouping items into small categories 

Tools I used in this competition are: numpy, pandas, sklearn, XGBoost GPU, LightGBM
All models are tuned on a windows10 with Intel i5 8thgen processor, 8GB RAM. Tuning models took about 12 to 16 hours, and training on the whole dataset took less than 15 minutes


# I. Exploratory Data Analysis
Information can be found in  2 EDA notebooks

# II. Feature Engineering
Information can be found in feature_eng  notebook

# III. Cross validations
Information can be found in function define within feature_eng  notebook called get_cv_idxs()

# IV. Training methods:
Information can be found in tuning_lgb and tuning_xgb notebook

# V. Ensembling
Information can be found in ensemble notebook


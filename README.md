# kaggle_predict_future_sales

In this competition i worked with a challenging time-series dataset consisting of daily sales data, provided by - 1C Company. The goal was to predict total sales for every product and store in the next month. 
By solving this competition i was able to apply and enhance my data science skills.
Main methods I used for this competition is LightGBM that provide the Leaderboard score: 0.90807(RMSE).

The most important features are:
* standard lag features (target value for X-months ago, target mean value for shop/item_category/items) (X=1,2,3,6,12)
* how much time this item is on the market (or is it brand new item that appear for the very first time)
* mean target values for new items (that appear for the very first time) for shop/item_category
* text feature  generated from shop_names 
* grouping items into small categories 

Tools I used in this competition are: numpy, pandas, sklearn, XGBoost GPU, LightGBM
All models are tuned on a windows10 with Intel i5 8thgen processor, 8GB RAM. Tuning models took about 12 to 16 hours, and training on the whole dataset took less than 15 minutes


# I. Exploratory Data Analysis
**Information can be found in  2 EDA notebooks**

* Take a quick look: 
   - Import each dataset and make a Quick exploration for each one 
   - Detect possible  outliers and missing values using (describe() and  hist() )
* Deeper investigation through some ambiguous or potential variables (Univariate Exploration): builds a solide foundation about the distribution and the structure of each dataset 
* Data cleaning:
   - item_cnt_day variable is heavily skewed, most of the values are arround 0 and 5.0 , this range contain 99.9% of the data range. but before droping them we need to check if those are a real outliers or special cases ( should check other items with item-id for those outlier items ).
   - my investigation lead me to drop some rows that are clearly outliers from their influence on the distribution of that specific item
   - i did the same for item_price variable 
   - for the shop dataset i found out that several shops are duplicates of each other (according to its name), so i fix the training and testings set
 * Discover and Visualize to gain insights: 
   - using summary statistics its hepls but not that much cause we're dealing with time-series datasets
   - so what realy helps is Exploratory Analysis(Bivariate Exploration specifically) using the date variable on the x-axis  and other variable on the y-axis
 * Data leakage:
     - several shops doesn't exist in the testset
     - investigate about the possibility of losing information if we drop those shops
     -  so as result of i drop some of them

# II. Feature Engineering
**Information can be found in feature_eng  notebook**

* Feature preprocessing and generation:
   - grouping items into small categories and apply one-hot-encoding to them   : generated from the first word in the item_category_name variable (with no Stopwords and with Stemming)
   - generate shop_city from shop_name and apply one-hot-encoding to them : detect the city using a list of all russian_cities
   - add sepecial dates: from  generate_calendar package i generate russian calendar that highlight contain 4 features [is_weekend	,is_business_day,	is_holiday] before adding those features to our dataset , i had to mean encoding them with the month variable.
   
* Feature extraction from text ( from shop_names and item_name) but unfortunately it doesn't improve the model
   - clean filter shops names and item_name before encoding them
   - Use TfidfVectorizer to transform item_name and shop_name into vectors.
   - Then use TruncatedSVD to reduce its dimensions to 10

* Mean encodings
   - Since the competition task is to make a monthly prediction, we need to aggregate the data to monthly level before doing any encodings.
   - the test set provided it contain just 2 columns ['shop_id', 'item_id'] and even new items that are note on the training data, so first we need to create a grid For every month('date_block_num') from all shops/items combinations from that month
   - then Add submission shop_id-item_id in order to test predictions #test['date_block_num'] = 34
   - generally mean encoding for all categorical features, Features encoded: item_id,shop_id,item_category_id and Target used for encoding: {'target':'sum','target_mean':np.mean}
   - finally, create standard lag features (target value for X-months(date_block_num) ago, target mean encoding value for shop/item_category/items) (X=1,2,3,6,12), using the created function **get_feature_matrix()**
   
* time features: the goal is to highlight how much time this item is on the market (or is it brand new item that appear for the very first time)
   - for this purpose i create two new features [ year_min, month_min] out of the date feature
   - then i generate a bunch of mean encoding for combinations of Features encoded:[ year_min, month_min, shop_id, item_category_id] and Target used for encoding: {'item_cnt_day':np.mean}
* finally we will end up this notebook by Train test split which is time based, and save them.  **now our data is prepared for a ML model**

# III. Cross validations
**Information can be found in function define within tuning_lgb notebook called get_cv_idxs()**

* Since we are dealing we a time series data so I have to pre-define which data can be used for train and test. I have a function called get_cv_idxs that will return a list of tuples for cross validation. I decide to use 6 folds, from date_block_num 28 to 33, and luckily this CV score is consistent to leaderboard score.

# IV. Training methods:
**Information can be found in tuning_lgb and tuning_xgb notebook**

* Metrics optimization: Regressors minimize mean squared error. Validation metric used RMSE, same as the evaluation metric of the project.

* Hyperparameter tuning:  using hyperopt package (early stopping ), then manually tune with GridSearchCVused  


* **1. LightGBM**
   when tuning the size of the tree, it’s better to tune min_data_in_leaf instead of max_depth. This means to let the tree grows freely until the condition for min_data_in_leaf is met. I believe this will allow deeper logic to develop without overfitting too much. Colsample_bytree and subsample are also used to control overfitting. And I keep the learning rate small throughout tuning.

* **2. XGBoost**
   I ran the XGBoost with CPU version, and I follow the same tuning procedures as mentioned. For some reason, I can’t seem to get a consistent result while running XGBoost, even with the same parameters. 


finally, I pick 2 models: one with max_depth tuned, and one without max_depth tuned, to get out-of-fold features and hoping they are different enough for ensembling.

# V. Ensembling
**Information can be found in ensemble notebook**

With XGB model, LightGBM-1 and LightGBM-2 out-of-fold features from previous methods, I calculated pairwise differences between them, get the mean of all 3, and include the most important features from feature importance: ‘target_lag_1’ to  the dataset.

From here I try few ensembling methods

Simple average and Weighted average
SKlearn linear regression and Elasticnet
Shallow Random Forest, tuned with 5 folds (from 29 to 33)
All of them results in RMSE score that is slightly better  than the LightGBM best model.



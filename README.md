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


* Feature extraction from text ( from shop_names and item_name) but unfortunately it doesn't improve the model
   - clean filter shops names and item_name before encoding them
   - Use TfidfVectorizer to transform item_name and shop_name into vectors.
   - Then use TruncatedSVD to reduce its dimensions to 10

* Mean encodings
   - Since the competition task is to make a monthly prediction, we need to aggregate the data to monthly level before doing any encodings.
   - the test set provided it contain just 2 columns to make ['shop_id', 'item_id'] and even new items that are note on the training data, so first we need to create a grid For every month('date_block_num') from all shops/items combinations from that month

Item counts for each shop-item pairs per month (‘target’). I also generated sum and mean of item counts for each shop per month(date_block_num) (‘shop_block_target_sum’,’shop_block_target_mean’), each item per month (‘item_block_target_sum’,’item_block_target_mean’, and each item category per month (‘item_cat_block_target_sum’,’item_cat_block_target_mean’)
   - standard lag features (target value for X-months ago, target mean value for shop/item_category/items) (X=1,2,3,6,12)
   - Generated mean encoding for all categorical features using expanding mean
   Features encoded: item_id,shop_id,item_category_id,month,year
   Target used for encoding: target, shop_target, item_target, category_target

* Prepare the data for ML algo

# III. Cross validations
**Information can be found in function define within feature_eng  notebook called get_cv_idxs()**
Train test split is time based.
Two ways to split for train and validation:
use last two month as validation set
Use date_block_num in {9,21,33} as validation set
After comparing the validation RMSE score vs. leaderboard RMSE score, selected the second validation method.

# IV. Training methods:
**Information can be found in tuning_lgb and tuning_xgb notebook**
Metrics optimization

Regressors minimize mean squared error. Validation metric used RMSE, same as the evaluation metric of the project.
Hyperparameter tuning

used early stopping to do parameter tuning for xgb and neural networks.

# V. Ensembling
**Information can be found in ensemble notebook**


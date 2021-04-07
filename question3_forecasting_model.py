import itertools
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

########Note= You can run this at command shell on the console  python question3_forecasting_model.py --dir "C:/Users/esra.ozalp/Documents/Case Study"(txt files location)

########Model Parameters
product_level_list = ['buscat', 'subcat', 'fineline','product']
feature_list = ['actual_sales_qty','actual_sales_amt']
product_dict = {'product':['fineline', 'subcat', 'buscat'],
                'fineline':['subcat', 'buscat'],  
                'subcat': ['buscat'], 
            }

def parse_arguments():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Directory that contains the input files')

    result = parser.parse_args()

    return result.dir


def read_csv(directory, file_name):
    file_path = os.path.join(directory, file_name)
    return pd.read_csv(file_path, delimiter=',')


def organizeTables(df):
    # Format column Names
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ','_')
    df.columns = df.columns.str.replace('[#%]','')
    df.columns = df.columns.str.strip()
    
    # Format product number
    df['product'] = df['product'].astype('str')
    df['product'] = df['product'].str.zfill(5)
    

def create445Calendar(year_num):
    # calculate  week-month dictionary eg: month[13]=3 
    month={}
    i=1
    m=12
    for m in range(1,m+1):
        if m%3==0:
          week_num = 5
        else:
          week_num = 4
        for w in range(i, i+week_num):
            month[w] = m
        i = i+week_num
    
    #create calendar table
    result=[]
    week_num = 52
    for y in range(1,year_num+1):
        for w in range(1,week_num+1):
            result.append([y,month[w],w])
            
    calendar = pd.DataFrame(result)
    calendar.columns = ['year','month','week']
    return calendar


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def createFullData(products, promo, actuals, forecast, calendar):
    product_list = list(pd.unique(products['product']))
    data = expand_grid({"year": list(range(1,5)), "week": list(range(1,53)), "product": product_list})

    df = pd.merge(data, calendar, how='left', on=['year','week'])
    df = pd.merge(df, promo, how='left', left_on=['year','week','product'], right_on=['deal_yr','deal_wk','product']).drop(['deal_yr','deal_wk', 'page', 'slot'], axis=1)
    df = pd.merge(df, products, how='left', on='product')
    df = pd.merge(df, actuals, how='left', left_on=['product','deal_num'], right_on=['product','deal_num']).drop(['deal_yr', 'deal_wk', 'page', 'slot'], axis=1)
    df = pd.merge(df, forecast, how='left', left_on=['product','deal_num'], right_on=['product','deal_num']).drop(['deal_yr', 'deal_wk'], axis=1)
    return df


def add_R52_deal_sales_old(product_level_list, feature_list, df_new):  
    for product_level in product_level_list:
        for feature in feature_list:
            weekly_product_sum = df_new.groupby([product_level,'year','week'])[feature].agg(['sum','count'])\
                        .rename(columns={'sum':'week_deal_sum','count':'week_deal_count'}).reset_index()
            weekly_product_sum[product_level + '_R52_' + feature + '_sum'] = weekly_product_sum.groupby(product_level).week_deal_sum\
                    .transform(lambda x: x.rolling(52).sum().shift())
            weekly_product_sum[product_level + '_R52_' + feature + '_count'] = weekly_product_sum.groupby(product_level).week_deal_count\
                    .transform(lambda x: x.rolling(52).sum().shift())
            weekly_product_sum[product_level + '_R52_' + feature + '_mean'] = weekly_product_sum[product_level + '_R52_' + feature + '_sum']/weekly_product_sum[product_level + '_R52_' + feature + '_count'] 

            df_new = pd.merge(df_new, weekly_product_sum, how='left', on=[product_level, 'year', 'week'])\
                    .drop(['week_deal_sum','week_deal_count'], axis=1)
    df_new = df_new.drop(['buscat_R52_actual_sales_amt_count',\
                           'subcat_R52_actual_sales_amt_count', \
                            'fineline_R52_actual_sales_amt_count',\
                            'product_R52_actual_sales_amt_count'], axis=1)
    return df_new


def createHierarchicalShares(df_new, feature_list, product_dict, week_num):
    for feature in feature_list:
        for item, values in product_dict.items():
            for value in values:
                df_new[item + '_' + value +'_share_' + feature + '_' + str(week_num)] = df_new[item + '_R' + str(week_num) + '_' + feature + '_sum']/df_new[value + '_R' + str(week_num) + '_' + feature +'_sum']
  

def historicalDealSales(data):
    df = data.groupby(['product','deal_num','year','week', 'deal_done']).actual_sales_qty.sum().reset_index()
    hist_data = data.groupby(['product','year','week', 'deal_done']).actual_sales_qty.sum().reset_index()\
                    .rename(columns={'year':'hist_year', 'week':'hist_week',\
                                     'actual_sales_qty':'hist_sales_qty', 'deal_done':'hist_deal_done'})


    agg_df = df.merge(hist_data, how='inner', on='product').query('hist_deal_done==1')
    agg_df['week_diff'] = agg_df['week'] - agg_df['hist_week']
    agg_df['year_diff'] = agg_df['year'] - agg_df['hist_year']
    agg_df['total_diff'] = agg_df['year_diff']*52 + agg_df['week_diff']
  

    # average sales quantity in 3-year historical data
    result = agg_df[ (agg_df.total_diff>=1) & (agg_df.total_diff<=156)]
    hist_df = result.groupby(['product', 'deal_num']).hist_sales_qty.agg([np.mean, np.max])\
          .rename(columns={'mean':'hist_sales_qty_avg', 'amax':'hist_avg_sales_max'})\
          .reset_index()

    # average sales quantity in last year the same period data
    ly_same_period = agg_df[ (agg_df.total_diff>=50) & (agg_df.total_diff<=54)]
    ly_same_period_df = ly_same_period.groupby(['product', 'deal_num']).hist_sales_qty.agg([np.mean, np.max])\
          .rename(columns={'mean':'ly_same_period_sales_qty_avg', 'amax':'ly_same_period_sales_qty_max'})\
          .reset_index()

    # previous dealsales qty avg and prev deal week diffrence
    prev_deal_diff = result.groupby(['product','deal_num']).total_diff.min()\
                    .reset_index()

    prev_df = pd.merge(result, prev_deal_diff, how='inner', on=['product', 'deal_num','total_diff'])
    prev_deal_df = prev_df.groupby(['product', 'deal_num', 'total_diff']).hist_sales_qty.agg([np.mean, np.max])\
          .rename(columns={'mean':'prev_deal_sales_qty_avg', 'total_diff':'prev_deal_week_diff', 'amax':'prev_deal_sales_qty_max'})\
          .reset_index()

    # merge new columns with master data
    data = pd.merge(data, hist_df, how='left', on=['product', 'deal_num'])
    data = pd.merge(data, ly_same_period_df, how='left', on=['product', 'deal_num'])
    data = pd.merge(data, prev_deal_df, how='left', on=['product', 'deal_num'])\
            .drop(['hist_avg_sales_max','ly_same_period_sales_qty_max','prev_deal_sales_qty_max'],axis=1)
    
    return data

def add_R52_deal_sales(product_level_list, feature_list, df_new, week_num):  
    for product_level in product_level_list:
        for feature in feature_list:
            weekly_product_sum = df_new.groupby([product_level,'year','week'])[feature].agg(['sum','count'])\
                        .rename(columns={'sum':'week_deal_sum','count':'week_deal_count'}).reset_index()
            weekly_product_sum[product_level + '_R' + str(week_num) +'_' + feature + '_sum'] = weekly_product_sum.groupby(product_level).week_deal_sum\
                    .transform(lambda x: x.rolling(week_num).sum().shift())
            weekly_product_sum[product_level + '_R' + str(week_num) +'_' + feature + '_count'] = weekly_product_sum.groupby(product_level).week_deal_count\
                    .transform(lambda x: x.rolling(week_num).sum().shift())
            weekly_product_sum[product_level + '_R' + str(week_num) +'_' + feature + '_mean'] = weekly_product_sum[product_level + '_R' + str(week_num) + '_' + feature + '_sum']/weekly_product_sum[product_level + '_R' + str(week_num) + '_' + feature + '_count'] 

            df_new = pd.merge(df_new, weekly_product_sum, how='left', on=[product_level, 'year', 'week'])\
                    .drop(['week_deal_sum','week_deal_count'], axis=1)
            df_new = df_new.drop([ #product_level + '_R' + str(week_num) +'_' + feature + '_sum',
                                  product_level + '_R' + str(week_num) +'_' + feature + '_count'], axis=1)
    return df_new


def main():
    directory = parse_arguments()

    ########## Read 4 txt data files
    # 1 - Actuals
    actuals = read_csv(directory, '01Actuals.txt')

    # 2 - Hierarchy
    products = read_csv(directory, '02Hierarchy.txt')

    # 3 - Promo Details
    promo = read_csv(directory, '03Promo Details.txt')

    # 4 - Promo Forecasts
    forecast = read_csv(directory, '04Promo Forecasts.txt')
        
    organizeTables(actuals)
    organizeTables(products)
    organizeTables(promo)
    organizeTables(forecast)
    calendar = create445Calendar(5)

    # calculate deal_duration day from deal_duration feature - have a numeric feature
    deal_day = promo.deal_duration.str.split().str[0].astype(int)
    deal_dur_type = promo.deal_duration.str.split().str[1].replace('Day','1').replace('Week','7').astype(int)
    promo['deal_duration_day'] = deal_day * deal_dur_type

    # Weekly 4-year sales data for all products in product table  4 X 52 X all products
    df_new = createFullData(products, promo, actuals, forecast, calendar)

    ###################### Data Engineering #######################################
    # Sales by rolling 52 week
    df_new = add_R52_deal_sales(product_level_list, feature_list, df_new, 52)
    createHierarchicalShares(df_new, feature_list, product_dict, 52)
    
    # Sales by rolling 26 week
    df_new = add_R52_deal_sales(product_level_list, feature_list, df_new, 26)
    createHierarchicalShares(df_new, feature_list, product_dict, 26)

    # Remove weeks that have no deals - turning to master data
    master = df_new[~df_new.sales_fcst_qty.isnull()]
    # Remove infinity numbers
    master = master.replace([np.inf, -np.inf], np.nan)


    # Create deal done flag
    predict_year = 4
    predict_week = 21
    master['deal_done']=1
    master.loc[(master.year==predict_year) & (master.week>=predict_week), 'deal_done']=0

    # Create historical features 
    master = historicalDealSales(master)

    # New features from categorical features
    '''
    # Convert position_in_flyer into is_frontback_cover
    master['is_frontback_cover'] = 0
    master.loc[master.position_in_flyer=='Front/Back Cover', 'is_frontback_cover'] = 1

    # Convert position_in_flyer into is_special_flyer
    master['is_special_flyer'] = 0
    master.loc[master.slot_description=='Special Flyer', 'is_special_flyer'] = 1

    # Convert deal_kind into Is_SPECIAL DEAL TYPE 2
    master['is_deal_type_second'] = 0
    master.loc[master.deal_kind=='SPECIAL DEAL TYPE 2', 'is_deal_type_second'] = 1
    '''
    # Week 47-16 have an increase-impact on sales by product
    master['is_special_week'] = 0
    master.loc[master.week.isin([47, 16]), 'is_special_week'] = 1

    ########FILL NULL VALUES#############
    # delete rows that have null values due to rolling
    delete_index = master[master['year']==1].index
    master = master.drop(delete_index, axis=0)

    # 1341 products has no historicl data. No deal before or a new product. Set -1
    master[master.hist_sales_qty_avg.isnull()] = master[master.hist_sales_qty_avg.isnull()].fillna(-1)

    # The rest of products have historical data but no sales in r52. 
    # Set 0 if numeric features. Categorical ones should set as N.
    master=master.fillna(0).fillna('N')

    ############# Drop unused columns
    drop_cols = ['actual_sales_amt', 'week', 'product', 'deal_num', 'year', 'fineline', 'subcat', 'buscat', 'sales_fcst_amt', 'deal_duration','line_of_business', 'month']
    master = master.drop(drop_cols, axis =1)

    ################# Encoding Categorical Variables
    categorical_features = list(master.select_dtypes(include=['object']).columns)
    for col in categorical_features:
        col = pd.get_dummies(master[col], prefix=col).astype('int')
        master = pd.concat([master,col],axis=1)
    master = master.drop(categorical_features, axis=1)

    ##################SPLIT DATA###########
    split_ratio = 0.7
    train_number = np.round(master.shape[0] * split_ratio).astype(int)
    data_train = master[:train_number]
    drop_cols = ['actual_sales_qty', 'sales_fcst_qty', 'deal_done']
    X_train = data_train.drop(drop_cols, axis=1)
    y_train = data_train['actual_sales_qty'].values

    data_test = master[train_number:]
    X_test = data_test.drop(drop_cols, axis=1)
    y_test = data_test['actual_sales_qty'].values
    y_test_forecasted = data_test['sales_fcst_qty'].values

    X_test_actual = data_test[data_test.deal_done==1].drop(drop_cols, axis=1)                           
    y_test_actual= data_test[data_test.deal_done==1]['actual_sales_qty'].values

    X_test_predict = data_test[data_test.deal_done==0].drop(drop_cols, axis=1)  

    ################ Modeling #################################
    ####Regression Forest######################################
    rf = RandomForestRegressor(random_state=42, n_jobs=-1 )
    params_rf = {'n_estimators' : [100,300, 500], 'max_depth':[None, 50, 100], 'min_samples_split':[5,10], 'min_samples_leaf':[5,10]}

    grid_search = GridSearchCV(rf, param_grid=params_rf, cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_, grid_search.best_score_)
    best_rf = grid_search.best_estimator_
    predict = best_rf.predict(X_test_actual)
    predict_train = best_rf.predict(X_train)
    predict_test = best_rf.predict(X_test)
    predict_test_predict = best_rf.predict(X_test_predict)

    print(f'Random Forest train\nMSE: {mean_squared_error(y_train,predict_train):.3f}\nMAE: {mean_absolute_error(y_train,predict_train):.3f}\nR2: {r2_score(y_train,predict_train):.3f}')
    print(f'Random Forest test\nMSE: {mean_squared_error(y_test_actual,predict):.3f}\nMAE: {mean_absolute_error(y_test_actual,predict):.3f}\nR2: {r2_score(y_test_actual,predict):.3f}')
    print(f'Random Forest Test Forecasted\nMSE: {mean_squared_error(y_test_forecasted,predict_test):.3f}\nMAE: {mean_absolute_error(y_test_forecasted,predict_test):.3f}\nR2: {r2_score(y_test_forecasted,predict_test):.3f}')

    predict_test_predict = best_rf.predict(X_test_predict)
    feature_importance_ = pd.DataFrame({'feature':X_train.columns, 'importance':best_rf.feature_importances_}).sort_values(by='importance',ascending=False)

    
    ####Gradient Boosted Tree######################################
    xgb = XGBRegressor(booster='gbtree', random_state=42)
    cv_params = {'max_depth': [10,50,None], 'min_child_weight': [3,5], 'n_estimators' : [300,500,1000]}

    grid_search_xgb = GridSearchCV(xgb,  param_grid = cv_params, scoring = 'r2', cv = 5, n_jobs = -1)

    print(grid_search_xgb.best_params_, grid_search_xgb.best_score_)
    best_xgb = grid_search_xgb.best_estimator_
    predict = best_xgb.predict(X_test_actual)
    predict_train = best_xgb.predict(X_train)
    predict_test = best_xgb.predict(X_test)
    print(f'Random Forest train\nMSE: {mean_squared_error(y_train,predict_train):.3f}\nMAE: {mean_absolute_error(y_train,predict_train):.3f}\nR2: {r2_score(y_train,predict_train):.3f}')
    print(f'Random Forest test\nMSE: {mean_squared_error(y_test_actual,predict):.3f}\nMAE: {mean_absolute_error(y_test_actual,predict):.3f}\nR2: {r2_score(y_test_actual,predict):.3f}')
    print(f'Random Forest Test Forecasted\nMSE: {mean_squared_error(y_test_forecasted,predict_test):.3f}\nMAE: {mean_absolute_error(y_test_forecasted,predict_test):.3f}\nR2: {r2_score(y_test_forecasted,predict_test):.3f}')

    predict_test_predict_xgb = best_xgb.predict(X_test_predict)
    feature_importance_xgb = pd.DataFrame({'feature':X_train.columns, 'importance':best_xgb.feature_importances_}).sort_values(by='importance',ascending=False).head(10) 


    ####SGD Linear Regression######################################
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_actual_scaled = scaler.transform(X_test_actual)
    X_test_scaled = scaler.transform(X_test)
    X_test_predict_scaled = scaler.transform(X_test_predict)

    lr = SGDRegressor(penalty='l2', max_iter=1000, random_state=42)
    param_grid = {'alpha':[1e-4, 1e-3,1e-5,3e-3,3e-4], 'eta0':[0.01, 0.1,0.3,0.03]}
    grid_search_lr = GridSearchCV(lr, param_grid=param_grid, cv=5, scoring='r2')
    grid_search_lr.fit(X_train_scaled, y_train)

    print(grid_search_lr.best_params_, grid_search_lr.best_score_)
    best_lr = grid_search_lr.best_estimator_
    predict = best_lr.predict(X_test_actual_scaled)
    predict_train = best_lr.predict(X_train_scaled)
    predict_test = best_lr.predict(X_test_scaled)

    predict_test_predict_xgb = best_xgb.predict(X_test_predict_scaled)

    print(f'SGD LINEAR REGRESSION Train\nMSE: {mean_squared_error(y_train,predict_train):.3f}\nMAE: {mean_absolute_error(y_train,predict_train):.3f}\nR2: {r2_score(y_train,predict_train):.3f}')
    print(f'SGD LINEAR REGRESSION Actual\nMSE: {mean_squared_error(y_test_actual,predict):.3f}\nMAE: {mean_absolute_error(y_test_actual,predict):.3f}\nR2: {r2_score(y_test_actual,predict):.3f}')
    print(f'SGD LINEAR REGRESSION Forecasted\nMSE: {mean_squared_error(y_test_forecasted,predict_test):.3f}\nMAE: {mean_absolute_error(y_test_forecasted,predict_test):.3f}\nR2: {r2_score(y_test_forecasted,predict_test):.3f}')

     ####Support Vector Regressor######################################
    svr = SVR()
    params_svr = [{'kernel':['linear'], 'C':[1e4,1e3,1e2,10,1,0.1,0.001,0.0001], 'epsilon':[0.1,0.001,1e-3,1e-4]},\
                {'kernel':[ 'rbf'],'gamma':['auto', 1e-2,1e-3,1e-4], 'C':[1e4,1e3,1e2,10,1,0.1,0.001,0.0001], 'epsilon':[0.1,0.001,1e-3,1e-4]}]

    grid_search_svr = GridSearchCV(svr, param_grid=params_svr, cv=5, scoring='r2')
    grid_search_svr.fit(X_train, y_train)

    print(grid_search_svr.best_params_, grid_search_svr.best_score_)
    best_svr = grid_search_svr.best_estimator_
    predict = best_svr.predict(X_test_actual_scaled)
    predict_train = best_svr.predict(X_train_scaled)
    predict_test = best_svr.predict(X_test_scaled)

    predict_test_predict_svr = best_svr.predict(X_test_predict_scaled)

    print(f'SVR Train\nMSE: {mean_squared_error(y_train,predict_train):.3f}\nMAE: {mean_absolute_error(y_train,predict_train):.3f}\nR2: {r2_score(y_train,predict_train):.3f}')
    print(f'SVR Actual\nMSE: {mean_squared_error(y_test_actual,predict):.3f}\nMAE: {mean_absolute_error(y_test_actual,predict):.3f}\nR2: {r2_score(y_test_actual,predict):.3f}')
    print(f'SVR Forecasted\nMSE: {mean_squared_error(y_test_forecasted,predict_test):.3f}\nMAE: {mean_absolute_error(y_test_forecasted,predict_test):.3f}\nR2: {r2_score(y_test_forecasted,predict_test):.3f}')


if __name__ == '__main__':
    main()
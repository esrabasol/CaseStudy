import pandas as pd
import numpy as np
import os
########Note= You can run this at command shell on the console  python question1_2_4.py --dir "C:/Users/esra.ozalp/Documents/Case Study"(txt files location)

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
    # map week-month & create a dictionary eg: month[13]=3 
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

################################ Question 1 ################################
def calculateCAGR(master, year_num):
    # Calculate CAGR for the first year_num years of the given data for the whole Line of Business
    sales = master[master.deal_yr<=year_num].groupby(['line_of_business','deal_yr']).actual_sales_amt.sum().reset_index()
    
    beginning_value = sales[sales.deal_yr==1].actual_sales_amt.values
    ending_value = sales[sales.deal_yr==year_num].actual_sales_amt.values
    
    cagr = ((ending_value/beginning_value)**(1/year_num))-1
    return np.round(cagr,2).item()


def getMonthMostAvgSales(master, year_num):
    # Calculate sales based on year & month
    year_month_sales = master[master.deal_yr<=year_num].groupby(['line_of_business','deal_yr','month'])\
                                                        .actual_sales_amt.sum().reset_index()\
                                                        .sort_values('actual_sales_amt', ascending=False)
    # Calculate monthly avg sales other than November or December
    monthly_avg_sales = year_month_sales[year_month_sales.month<=10].groupby(['month']).actual_sales_amt.mean().reset_index()
    monthly_avg_sales = monthly_avg_sales.sort_values('actual_sales_amt', ascending=False).head(1)

    return monthly_avg_sales.month.item(), int(round(monthly_avg_sales.actual_sales_amt.item(),0))

################################ Question 2 ################################
def month_by_month_sales(master, buscat):
    master['actual_sales_amt'] = master['actual_sales_amt'].fillna(master.sales_fcst_amt)
    buscat7_year_month_sales = master[master.buscat==buscat].groupby(['deal_yr','month']).actual_sales_amt.sum().reset_index()
    return buscat7_year_month_sales

def full_year_sales(df, year_num):
    full_year = df[df.deal_yr==year_num].actual_sales_amt.sum()
    return int(round(full_year,2).item())

################################ Question 4 ################################
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

def parse_arguments():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Directory that contains the input files')

    result = parser.parse_args()

    return result.dir


def read_csv(directory, file_name):
    file_path = os.path.join(directory, file_name)
    return pd.read_csv(file_path, delimiter=',')


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
    
    # header format & fix Product format
    organizeTables(actuals)
    organizeTables(products)
    organizeTables(promo)
    organizeTables(forecast)
    calendar = create445Calendar(4)

    # Create a master data table
    master = pd.merge(promo, products, how='inner', on='product')
    master = pd.merge(master, actuals.drop(['deal_yr', 'deal_wk', 'page', 'slot'], axis=1), how='left', on=['product','deal_num'])
    master = pd.merge(master, forecast.drop(['deal_yr', 'deal_wk', 'page', 'slot'], axis=1), how='left', on=['product','deal_num'])
    master = pd.merge(master, calendar, how='left', left_on=['deal_yr','deal_wk'], right_on=['year','week']).drop(['year','week'], axis=1)

    # Question 1.1 Calculate CAGR for the first three years of data for the whole Line of Business
    cagr = calculateCAGR(master, 3)
    print('CAGR: ', cagr)

    # Question 1.2 Besides November or December, which month generates the most sales dollars on average? 
    month_with_most_sales, sale_avg = getMonthMostAvgSales(master, 3)
    print('The month with most sales was {} with average amount ${}'.format(month_with_most_sales, sale_avg))

    # Question 2.1 For BUSCAT07, show the month-by-month sales in $ for all four years (note: for any timeframes without recorded actuals, estimate the sales using the existing forecasts provided).
    buscat7_year_month_sales = month_by_month_sales(master, 'BUSCAT07')
    
    # Question 2.2 With precision down to the dollar, what do you estimate the full year 04 sales to be for BUSCAT07? 
    year_4_sales = full_year_sales(buscat7_year_month_sales, 4)

    print('Estimated the full year 04 sales to be for BUSCAT07 is ${}'.format(year_4_sales))

    # Question 4.1 What sales uplift is expected when a product in the flyer is also featured on TV
    # read Tv indicator table
    tv_indicator =  read_csv(directory, 'TV Indicator.txt', delimiter=',', encoding = "utf-16")
    organizeTables(tv_indicator)

    # get product list
    tv_products = list(pd.unique(tv_indicator[product]))

    # get historical data
    master = master.rename(columns={deal_yr:year, deal_wk:week})
    predict_year = 4
    predict_week = 21
    master[deal_done]=1
    master.loc[(master.year==predict_year) & (master.week>=predict_week), deal_done]=0

    tv_products_full_data = master[master[product].isin(tv_products)]
    tv_products_full_data = historicalDealSales(tv_products_full_data)

    # select only related weeks
    tv_data = pd.merge(tv_products_full_data, tv_indicator, how = inner, left_on = [product, year, week], right_on = [product, deal_yr, deal_wk])

    cols = [product,deal_yr, deal_wk, promo_price, hist_sales_qty_avg, actual_sales_qty, actual_sales_amt, sales_fcst_qty, sales_fcst_amt]
    tv_data = tv_data[cols]
    tv_data[profit] = tv_data.actual_sales_qty * tv_data.promo_price * 0.3 

    # Question 4.2  If the cost of making a commercial is $1MM per product and the typical product makes 30% margin, was it beneficial for the company to make those commercials? 
    tv_data['gross_profit'] = tv_data.actual_sales_qty * tv_data.promo_price * 0.3 
    total_profit = tv_data.groupby('product').gross_profit.sum().reset_index()

    total_profit['commercial_cost'] = 1000000.00
    total_profit['net_profit'] = total_profit['gross_profit'] - total_profit['commercial_cost']
    company_profit = round(total_profit.net_profit.sum(),2)

    print('The company has got ${} profit from those products.'.format(company_profit))

if __name__ == '__main__':
    main()

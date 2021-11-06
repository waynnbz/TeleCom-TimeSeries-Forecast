import pandas as pd
import json
import pickle
import argparse
from datetime import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import datetime
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

grp_cols1 = ['Leopard Closing Base', 'Leopard Leavers', 
            'Panther Closing Base', 'Panther Leavers', 
            'Hyena Closing Base', 'Hyena Leavers']
grp_cols2 = ['Panther Gross Adds']
grp_cols3 = ['Leopard Gross Adds', 'Hyena Gross Adds', 'Panther - Leopard - Hyena Total Revenue']

def read_config(config_filepath):
    
    with open(config_filepath, "r") as f:
        config = json.load(f)
    
    run_config_sanity_check(config)
    return(config)

def run_config_sanity_check(config):
    
    assert pd.to_datetime(config['start-date'], format = '%Y-%m-%d'), "Start-date is not in proper format(%Y-%m-%d)"
    assert pd.to_datetime(config['end-date'], format = '%Y-%m-%d'), "End-date is not in proper format(%Y-%m-%d)"
    
    start_date = pd.to_datetime(config['start-date'], format = '%Y-%m-%d')
    end_state = pd.to_datetime(config['end-date'], format = '%Y-%m-%d')
    
    assert end_state >= start_date, "Start-date is after End-date"
    
    assert config['forecast-length']>0, "Forecast length must be an integer and greater than zero"
    assert config['num-iterations']>0, "Number of Iterations must be an integer and greater than zero"


def read_dataset(filepath):
    
    data = pd.read_excel(filepath)
    cols = data.iloc[1,:]
    data = data.iloc[2:,:]
    data.columns = cols
    variables = (data['Generic Product ']+" "+data['Generic Variable']).values.tolist()
    processed_data = pd.DataFrame(data.iloc[:, 9:].values.T.astype('float'), columns = variables, index = data.columns[9:])
    processed_data.index = pd.to_datetime(processed_data.index)
    processed_data.index.name = 'date'
    
    return(processed_data)
    
    
def prepare_dataset(data, start_date, end_date):
    
    train = data[data.index<start_date]
    test = data[(data.index>=start_date) & (data.index<=end_date)]
    print(train.shape, test.shape)
    return(train, test)
    
def remove_outliers(series):
    
    median = series.median()
    diff = np.abs(series - median) 
    mad = diff.median()
    mod_z = 0.6745*diff/mad
    series.loc[mod_z>3.5] = np.nan
    series = series.fillna(method = "bfill")
    return(series)
    
def read_params():
    
    with open("params.pkl", "rb") as f:
        params = pickle.load(f)
    return(params)


def fit_predict(params, train_series, forecast_length):
    
    # MAD 
    train_series = remove_outliers(train_series)
    my_order, my_seasonal_order = params[0:3], params[3:]
    
    model = SARIMAX(train_series, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(forecast_length)
    
    return(yhat)
        
def build_and_predict(train, test, config):
    
    params = read_params()
    print(params)
    preds = test.copy()
    
    for col in train.columns:
        preds[col] = fit_predict(params[col], train[col], config['forecast-length'])
            
    return(preds)

def main(args):
    
    config = read_config(args.configfile)
    
    #read the dataset
    cols = ['Falcon Average revenue per new customer - Falcon',
        'Falcon Average revenue per existing customer - Falcon',
        'Falcon Gross Adds - Falcon(Norm)',
        'Falcon Net Migrations - Falcon(Norm)']
    data = read_dataset(args.filepath)[cols]
    
    start_date = dt.strptime(config['start-date'], "%Y-%m-%d")
    end_date = dt.strptime(config['end-date'], "%Y-%m-%d")
    
    # prepare the datasset the for the iteration
    train, test = prepare_dataset( data, start_date, end_date)
        
    # create the model and predict
    preds = build_and_predict(train, test, config)
        
    # save the submission in robust.
    preds.to_csv("final.csv")
        
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", type=str, default = "config.json", 
                        help = "path to configuration file")
    parser.add_argument("--filepath", type=str, default = "data.xlsx", 
                        help = "path to data xlsx file")
    
    args = parser.parse_args()
    main(args)
    
    
    
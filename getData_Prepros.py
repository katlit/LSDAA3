import json
import pandas as pd
from sklearn.model_selection import train_test_split

def get_train_test(json_filename):
    #Get data
    dataframe = pd.read_json (json_filename, orient = "split")
    
    #Clean data
    dataframe = dataframe.dropna(axis=0)
    dataframe = dataframe.drop(["Lead_hours", "Source_time", "ANM", "Non-ANM"], axis=1)
    
    """
    Sets the 'total' column in a given df x to be an average of the values 
    in the interval [i-t_delta: i+t_delta]
    renames the column to Avg_total
    """ 
    t_delta = pd.Timedelta("1 hour 30 minutes") #Specify time for average time frame
    
    for i in dataframe.index:
        dataframe.loc[i, "Total"] = dataframe.loc[i - t_delta: i + t_delta, "Total"].mean(skipna = True)
    
    dataframe = dataframe.rename(columns = {"Total": "Avg_total"}) #Recast total to be an average over a time span given by t_delta

    '''
    Create traning, validation and test set
    '''
    y = dataframe["Avg_total"]
    x = dataframe.drop("Avg_total", axis = 1)
    
    x_dev, x_test, y_dev, y_test = train_test_split(x, y, test_size = 1/5, shuffle = False) #20% of total data is test data
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size = 1/4, shuffle = False) #20% of total data is val data
    
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_test("dataset.json") 
    print(x_train, y_train)
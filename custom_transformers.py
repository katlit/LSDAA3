from math import pi
from numpy import arange
import numpy as np

import datetime
from sklearn.base import BaseEstimator, TransformerMixin

class Data_shaper(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x, y=None):
        x = x.drop(["Lead_hours", "Source_time"], axis=1)
        x = x.dropna(axis = 0)
        return x

class Convert_direction(BaseEstimator, TransformerMixin):
    """
    Takes input of wind_df, returns df with new column "Direction" in radians
    """
    def fit(self, x, y=None):
        return self
    def transform(self, x, y=None):
        directions =  ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
        direct_dict = {directions[i]: pi/8*i for i in range(len(directions))} # pi/8 = 1/16 turn in radian
        #print(direct_dict)
        x["Degrees"] = x["Direction"].map(direct_dict) #Add Degrees table to df
        x = x.drop("Direction", axis = 1)
        #print(x)
        return x


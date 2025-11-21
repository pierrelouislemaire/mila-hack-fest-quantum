import numpy as np
import data as utils
import scipy
import math

def copypaste_context(context: np.array, forecast_lenght: int) -> np.array:
    
    context_lenght = context.shape[0]
    repeat_times = math.ceil(forecast_lenght / context_lenght)
    
    forecast = np.tile(context, (repeat_times, 1))[:forecast_lenght]
    
    return forecast
    
def mean_context(context: np.array, forecast_lenght) -> np.array:
    
    mean_context = np.mean(context, axis=0)
    forecast = np.tile(mean_context, (forecast_lenght, 1))
    
    return forecast

def harmonic_mean_context(context: np.array, forecast_lenght) -> np.array:
    
    mean_context = np.mean(context, axis=0)
    forecast = np.tile(mean_context, (forecast_lenght, 1))
    
    return forecast 
    
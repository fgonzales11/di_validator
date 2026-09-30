
import pandas as pd
import math
# Simple lookup dictionary for pandas frequencies to fractions of an hour.



def InferFreq(timestamps):
    freq = pd.infer_freq(pd.to_datetime(timestamps))
    return freq

def ToAvgPower(df, channels=['consumption', 'generation'], override_freq=None):
    freq_dict = {'5T': 5.0/60, '15T':0.25 ,'30T': 0.5, 'H': 1,'2H':2, '4H': 4 }
    #df.reset_index(inplace=True)

    if override_freq == None:
        map = df.reset_index().groupby('DeviceID')['ValueDate'].apply(InferFreq).to_dict()                
    #df.reset_index(inplace=True)
        df['frequency'] = df.index.droplevel(1).map(map)            
    else:
        df['frequency'] = override_freq        

    df['division_factor'] = df['frequency'].map(freq_dict)    
    for channel in channels:
        df['AvgPower_'+channel] = df[channel]/df['division_factor']
    
    return df.drop(columns=['frequency', 'division_factor'])

def Resample(intv, rs='H', agg = {'consumption':'sum', 'generation': 'sum', 'voltage':'mean'}):
    return intv.reset_index('DeviceID').groupby('DeviceID').resample(rs).agg(agg)
    

def ToChangePoint(intv, channels=['consumption', 'generation', 'voltage']):
    
    #calculate the changepoint value for each column
    for c in channels:                        
        intv['{}_diff'.format(c)] = intv[c] - intv[c].shift(1)    
    
    return intv


#apply to dataframe axis =1
def PowerFactor_KVA(row):
    
    kVA = math.sqrt(row['consumption']**2+row['reactive']**2)
    row['kVA'] = kVA
    if kVA > 0:
        row['PowerFactor'] = row['consumption']/row['kVA']

    return row
import pandas as pd
from abc import ABC, abstractmethod
import ModelManager
from Transformations import IntervalTransformations as itr

#Abstract Template that interval based predictors should follow.
class TemplatePredictor(ABC):
    
    #Returns dictionary of loaded models
    def LoadModel(self, model, isNeuralNet=False):
        print('loading specific ML model: {}'.format(model))
        return ModelManager.LoadModel(model, isNeuralNet=isNeuralNet)

    #Predictors have a prepration phase and Prediction Phase 
    @abstractmethod
    def Prepare(self, raw_batch):
        pass
    @abstractmethod
    def Predict(self, prepared_batch):
        pass

    def Execute(self, raw_batch):

        prepared_batch = self.Prepare(raw_batch)
        prepared_batch = self.Predict(prepared_batch)
        return prepared_batch[['prob']] #make sure to return a dataframe and not a series


class EVPredictor(TemplatePredictor):
    def __init__(self, models=['EV_transform1_MinMaxScaler_30Min','EV_transform2_PCA_30Min', 'EV_Gauss_M3only_30Min'], LoadData=False):        
                
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn import decomposition
        from sklearn import preprocessing
        from sklearn import gaussian_process
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        
        self.Transfomer1 = self.LoadModel(models[0])
        self.Transfomer2 = self.LoadModel(models[1])
        self.Predictor = self.LoadModel(models[2])

        if LoadData:
            self.TrainingData = self.LoadModel('EV_Gauss_Data')

        #Define Daytime Buckets on init
        night_hours =[0,1,2,3,4,23]
        evening_hours = [17,18,19,20,21,22]
        morning_hours = [5,6,7,8,9,10]
        day_hours = [11,12,13,14,15,16]
        self.hour_map = {hour: 'night' for hour in night_hours}
        self.hour_map.update({hour:'evening' for hour in evening_hours})
        self.hour_map.update({hour:'morning' for hour in morning_hours})
        self.hour_map.update({hour:'day' for hour in day_hours})                


    def Transform(self, X):
        return self.Transfomer2.transform(self.Transfomer1.transform(X))              

    def Prepare(self, raw_batch, additional_index=[]):
        
        #raw_batch = itr.Resample(raw_batch,rs='H', agg={'consumption':'sum'})
        raw_batch = itr.ToAvgPower(raw_batch, channels=['consumption'], override_freq='30T')
        raw_batch = itr.ToChangePoint(raw_batch, channels=['AvgPower_consumption'])

        raw_batch['H'] = raw_batch.index.droplevel(0).hour #Assign hour to each interval
        raw_batch['W'] = raw_batch.index.droplevel(0).week #Assign week of year to each interval
        raw_batch['Y'] = raw_batch.index.droplevel(0).year #Assign year to each interval                                              
        raw_batch['grp'] = raw_batch['H'].map(self.hour_map) #map interval to bucket
        raw_batch['grp'].fillna('day', inplace=True)

        #index = ['DeviceID','W','Y', 'grp'].extend(additional_index)     
        # aggregate stats for each device/label/bucket
        agg = raw_batch.reset_index('ValueDate').groupby(['DeviceID','W','Y','grp']).agg({'AvgPower_consumption_diff':['mean','max','min','std']}).fillna(0)
        
        agg.reset_index('grp')
        prepared_batch = pd.pivot_table(agg.sort_index(), index=['DeviceID','Y','W'], values=['AvgPower_consumption_diff'], columns=['grp']).fillna(0)
        
        return prepared_batch['AvgPower_consumption_diff']


    def Predict(self, prepared_batch):                

        X = prepared_batch.values

        Xf = self.Transform(X)

        prepared_batch['prob'] = self.Predictor.predict_proba(Xf)[:,1]                
        
        return prepared_batch



class SolarPredictor(TemplatePredictor):
    
    def __init__(self, model_name='knn_solar_v2_skl0-22-2-post'):        
        self.clf = self.LoadModel(model_name, isNeuralNet=False)    

    def Prepare(self, raw_batch):
        #Get interval all interval data for identified solar customers.

        #Resample to hourly data so it is applicable to multiple customers
        raw_batch['hour'] = raw_batch.droplevel(0).index.hour
        raw_batch['week']  = raw_batch.droplevel(0).index.week
        raw_batch['year'] = raw_batch.droplevel(0).index.year
        raw_batch['minute'] = raw_batch.droplevel(0).index.minute
        
        peak_intvl = raw_batch.reset_index()[['DeviceID','week','year','hour','minute','generation']].groupby(['DeviceID','week','year','hour','minute']).max().reset_index()    
        #Find peak hour generation for each week, (e.g. peak 1 p.m. generation for the week)

        #Get weekly total generation 
        theory_peak_hour = peak_intvl.groupby(['DeviceID','year','week','hour']).sum().reset_index()

        gen_profile = pd.pivot_table(theory_peak_hour,index = ['DeviceID','year','week',],values =['generation'], columns=['hour']).droplevel('hour', axis=1).dropna()

        #Divide each hour by the sum (normalize row vector)
        prepared_batch = gen_profile.div(gen_profile.sum(axis=1), axis=0).fillna(0)        
        return prepared_batch

    def Predict(self, prepared_batch):
        
        X = prepared_batch.values
        prepared_batch['prob'] = self.clf.predict_proba(X)[:,1]  
        #return index and label (DeviceID in index)
        return prepared_batch



class NN_EV(TemplatePredictor):
    def __init__(self):
        self.model = self.LoadModel('NN_EV', isNeuralNet=True)
            
    def Prepare(self, raw_batch):

        raw_batch['Y'] = raw_batch.index.droplevel(0).year
        raw_batch['W'] = raw_batch.index.droplevel(0).week
        raw_batch = itr.ToAvgPower(raw_batch, channels=['consumption'], override_freq='30T')

        raw_batch = itr.ToChangePoint(raw_batch, channels=['AvgPower_consumption'])

        raw_batch['AvgPower_consumption_diff'] = raw_batch['AvgPower_consumption_diff']/30 #hard coded scaling factor for now

        raw_batch = raw_batch.reset_index().set_index(['DeviceID', 'Y', 'W'])

        raw_batch['rank'] = raw_batch.groupby(raw_batch.index)['ValueDate'].rank(method="first", ascending=True)       
        raw_batch = raw_batch[raw_batch['rank'] <= 336] #End of the calendar year but 1st week of iso year causing issues

        prepared_batch = pd.pivot_table(raw_batch.reset_index(), index=['DeviceID', 'Y', 'W'], columns='rank', values='AvgPower_consumption_diff')
        prepared_batch.fillna(0, inplace=True)                        
        return prepared_batch

    def Predict(self, prepared_batch):

        X = prepared_batch.values.reshape(len(prepared_batch),336, 1)
        prepared_batch['prob'] = self.model.predict(X).flatten() #need to flatten neural network outputs        
        return prepared_batch



class NN_EV_wTemp(TemplatePredictor):
    def __init__(self, model_name='NN_EV_wTemp_20200930'):
        self.model = self.LoadModel(model_name, isNeuralNet=True)          

    def Prepare(self, raw_batch):                
        
        # Should have Consumption and intervals
        raw_batch['year'] = raw_batch.index.droplevel(0).year
        raw_batch['week'] = raw_batch.index.droplevel(0).week
        raw_batch = itr.ToAvgPower(raw_batch, channels=['consumption'], override_freq='30T')
        raw_batch = itr.ToChangePoint(raw_batch, channels=['AvgPower_consumption'])
        raw_batch['AvgPower_consumption_diff'] = raw_batch['AvgPower_consumption_diff']/30 #hard coded scaling factor for now
        
        
        raw_batch['Temp'] = (raw_batch['Temp']+60)/200 #scale for extreme weather by adding 60 (for that -60F day) and dividing by 200                    
        raw_batch['Temp'].interpolate(inplace=True) #interpolate any gaps in the weather

        raw_batch = raw_batch.reset_index().set_index(['DeviceID', 'year', 'week'])
        raw_batch['rank'] = raw_batch.groupby(raw_batch.index)['ValueDate'].rank(method="first", ascending=True)       
        raw_batch = raw_batch[raw_batch['rank'] <= 336] #End of the calendar year but 1st week of iso year causing issues


        data = pd.pivot_table(raw_batch.reset_index(), index=['DeviceID','year', 'week'], columns='rank', values=['AvgPower_consumption_diff', 'Temp'])

        data.fillna(0, inplace=True)                
        #Seperate Temps from Consumption, sort index and reshape
        temps2 = data[['Temp']].copy(deep=True)
        temps2.columns = temps2.columns.droplevel()
        temps2['channel'] = 'Temp'

        pwr = data[['AvgPower_consumption_diff']].copy(deep=True)
        pwr.columns = pwr.columns.droplevel()
        pwr['channel'] = 'consumption'
        data2 = pd.concat([pwr, temps2])
        data2 = data2.reset_index().set_index(['DeviceID', 'year', 'week','channel'])
        
        
        prepared_batch = data2.sort_index()        
        return prepared_batch

    def Predict(self, prepared_batch):
        
        X = prepared_batch.values.reshape(int(len(prepared_batch)/2), 336,2)
        
        index = [val[0:3] for val in prepared_batch.index.values if val[3] =='Temp']
        probs = self.model.predict(X).flatten() #need to flatten neural network outputs        
        ridx = pd.MultiIndex.from_tuples(index, names = ['DeviceID', 'year', 'week'])
        results = pd.DataFrame(data={'prob':probs}, index=ridx)

        return results



class EnsemblePredictor():
    def __init__(self, models=['EVPredictor', 'NN_EV']):
                
        fac = PredictorFactory()
        self.clfs = {}
        for pred in models:
            self.clfs[pred] = fac.create_predictor(pred)
        return

    def Execute(self, raw_batch):
        frames = []
        for modelname, model in self.clfs.items():
            model_pred = model.Execute(raw_batch)            
            model_pred.columns = pd.MultiIndex.from_tuples([(modelname, col[0]) for col in model_pred.columns.values])
            frames.append(model_pred)

        #Combine results of sub models into one dataframe and sort column index        
        labels = pd.concat(frames, axis=1)
        labels.sort_index(axis=1, inplace=True)        
        
        return labels

#Class to Return Predictors
class PredictorFactory():
    def create_predictor(self, name, **kwds):
        #Exact name of the predictor in     
        print('loading predictor template: {}'.format(name))
        
        if name == 'EVPredictor':
            print('loading EV')
            return EVPredictor(**kwds)
        if name == 'Solar-KNN':
            return SolarPredictor(**kwds)
        if name == 'NN_EV':
            return NN_EV(**kwds)
        if name == 'NN_EV_wTemp':
            return NN_EV_wTemp(**kwds)

        if name == 'Ensemble':
            return EnsemblePredictor(**kwds)
        #print(d[targetpredictor])
        else:
            print('no predictor found')

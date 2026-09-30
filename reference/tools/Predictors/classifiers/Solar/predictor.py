'''import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime, timedelta
from OO import OODataHelper as odh
import helpers
import config
import os

class Predictor(odh.OODataHelper):
    def __init__(self, server, db,args):        
        
        odh.OODataHelper.__init__(self, server=server, db=db)
        self.clf = helpers.LoadModel('knn_solar_v2_skl0-22-2-post')

        self.mode = args[0]
        if self.mode == 'latest':        
            n_days = int(args[1])
            datetime.utcfromtimestamp
            self.end = pd.to_datetime(self.OO_GetAMIDataStats()['LastRecRegTime'].values[0])                        
            self.start = (self.end + pd.Timedelta(days=-n_days)).strftime('%Y-%m-%d')
            self.end = self.end.strftime('%Y-%m-%d')
            self.kwh_thresh = args[2]                        
            self.dest_table = args[3].split('.')[1]
            self.dest_schema =args[3].split('.')[0]            

            print('configured for latest data predictions on generation above {} between the dates {} to {}'.format(self.kwh_thresh, self.start, self.end))
        if self.mode == 'daterange':
            print('configuring predictor for date range predictions')
            self.start = args[1]
            self.end = args[2]
            self.kwh_thresh = args[3]
            self.dest_table = args[4].split('.')[1]
            self.dest_schema =args[4].split('.')[0]       
            print('configured for date range predictions on generation above {} between the dates {} to {}'.format(self.kwh_thresh, self.start, self.end))
        if self.mode == 'table':
            print('getting custom configuration table')
            self.input_table = args[1]
            self.dest_table = args[2].split('.')[1]
            self.dest_schema =args[2].split('.')[0]
        return
    
    def GenerateBatches(self):
        df = self.OO_ExecuteSQL('Select deviceid, start, [end] from {}'.format(self.input_table))    
        
        df['start'] = pd.to_datetime(df['start']).dt.strftime('%Y-%m-%d')
        df['end'] = pd.to_datetime(df['end']).dt.strftime('%Y-%m-%d')
        
        return df

    # Look at start and end registers for generation channel.            
    def GenerateLeads(self):
        gen_channelids = self.OO_GetChannelIDs(['generation'],'register').ChannelID.values
        
        sql = open(config.root+'/src/Predictors/classifiers/Solar/leads.sql').read().format(startdate=self.start,enddate=self.end, generation_reg=','.join(str(c) for c in gen_channelids), kwh_thresh=self.kwh_thresh)    
        df = self.OO_ExecuteSQL(sql)
        return df
    

    def Transform_Day(self, df):
        df['day'] = df.index.day.values
        df['hour'] = df.index.hour.values   
        df['minute'] = df.index.minute.values
    
        #Find peak hour generation for each week, (e.g. peak 1 p.m. generation for the week)
        peak_intvl = df[['deviceid','hour','minute','generation']].groupby(['deviceid','day','hour','minute']).max().reset_index()    
        theory_peak_hour = peak_intvl.groupby(['deviceid','day','hour']).sum().reset_index()

        #Normalize usage profile
        gen_profile = pd.pivot_table(theory_peak_hour,index = ['deviceid','day'],values =['generation'], columns=['hour']).droplevel('hour', axis=1).dropna()
   
        gen_profile = gen_profile.div(gen_profile.sum(axis=1), axis=0).fillna(0)
        
    def Transform(self,df):
        #Get interval all interval data for identified solar customers.

        #Resample to hourly data so it is applicable to multiple customers
        df['hour'] = df.index.hour.values   
        df['minute'] = df.index.minute.values
    
        #Find peak hour generation for each week, (e.g. peak 1 p.m. generation for the week)
        peak_intvl = df[['deviceid','hour','minute','generation']].groupby(['deviceid','hour','minute']).max().reset_index()    
        theory_peak_hour = peak_intvl.groupby(['deviceid','hour']).sum().reset_index()

        #Normalize usage profile
        gen_profile = pd.pivot_table(theory_peak_hour,index = ['deviceid'],values =['generation'], columns=['hour']).droplevel('hour', axis=1).dropna()
   
        gen_profile = gen_profile.div(gen_profile.sum(axis=1), axis=0).fillna(0)
        
        return gen_profile

    def Predict(self, prepped_df):

        X = prepped_df.values

        #Get probability of device readings indicating solar
        prepped_df['solar_prob'] = self.clf.predict_proba(X)[:,1]

        #return index and label (deviceid in index)
        return prepped_df['solar_prob']
                    
    def ExecuteBatch(self, devices, start_dates, end_dates):

        intv = self.OO_GetIntervalData(devices, start_dates, end_dates, channels=['generation'])  

        t = self.Transform(intv)
        
        labels = self.Predict(t) 

        self.OO_ImportDataframe(labels, self.dest_table,  schemaname=self.dest_schema, if_exists='append')


    def Execute (self, batch_size = 10):
        print('Starting Execution')
        if self.mode == 'table':
            df = self.GenerateBatches()
            
            n_devices = len(df.deviceid.values)
            for i in range (0, n_devices, batch_size):  
                tmp = None   
                if i + batch_size < n_devices:           
                    tmp = df.iloc[i:i+batch_size]                                                            
                else:
                    tmp = df.iloc[i:i+batch_size]                                
                self.ExecuteBatch(tmp['deviceid'].values,tmp['start'].values,tmp['end'].values )

        if self.mode == 'latest' or self.mode == 'daterange':
            devices = self.GenerateLeads().deviceid.values
            n_devices = len(devices)
            for i in range (0, n_devices, batch_size):  
            
                if i + batch_size < n_devices:                                                   
                    self.ExecuteBatch(devices[i:i+batch_size], self.start, self.end) 
                else:
                    self.ExecuteBatch(devices[i:], self.start, self.end)                 

            if self.mode =='latest':
 
                merge_sql_raw = open(config.root+'/src/Predictors/classifiers/Solar/merge.sql').read()
                merge_sql = merge_sql_raw.format(destination_table='.'.join([self.dest_schema,self.dest_table]), startdate=self.start, enddate =self.end)

                self.OO_ExecuteSQL(merge_sql)

        if self.mode == 'inception':
            df = self.OO_ExecuteSQL('select distinct deviceid from project.SolarSP')
            template_sql = open(config.root + '/src/Predictors/classifiers/Solar/inception.sql').read()
            gen_channelids = self.OO_GetChannelIDs(['generation'],type='register').ChannelID.values
            channels = ','.join([str(c) for c in gen_channelids])
            n_devices = len(df.deviceid.values)
            
            for i in range (0, n_devices, batch_size):  
                tmp = None   
                if i + batch_size < n_devices:           
                    tmp = df.iloc[i:i+batch_size]                                                            
                else:
                    tmp = df.iloc[i:i+batch_size]                               

                devices = ','.join([str(d) for d in tmp['deviceid'].values])                
                sql = template_sql.format(devices=devices, channels=channels,kwh_thresh=0.1,d=10)

                devices_dates = self.OO_ExecuteSQL(sql)

                dates = pd.to_datetime(devices_dates['valuedate']).dt.strftime('%Y-%m-%d').values
                batch_intervals =  self.OO_GetIntervalData(devices=devices_dates.deviceid.values, start_dates=dates,end_dates=dates, channels=['generation'] )

                t = self.Transform_Day(batch_intervals)
                
                results = self.Predict(t)

                self.OO_ImportDataframe(results, 'Solar_Inception_Test', 'project', if_exists='append')'''

                
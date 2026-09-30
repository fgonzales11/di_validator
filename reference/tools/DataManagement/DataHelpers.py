
import pandas as pd

import sqlalchemy as s
import numpy as np
import os
from datetime import datetime, timedelta 
from DataManagement import config  #There is probably a better 
import sys
import math
import pymssql

#Abstract class for handling general SQL execution
class DBConnection():

    def __init__(self, server, db):
        self._engine = None
        self.server = server
        self.db = db
        #self._session_factory = None
        #self._Session = None
    
    @property    
    def engine(self):
        return self._engine       
    
    @engine.setter
    def engine(self, value):
        self._engine = value    


    def LoadSQLFile(self, filename):
        subd = 'SQLFiles/'
        # determine if application is a script file or frozen exe

        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the PyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app 
            # path into variable _MEIPASS'.
            application_path = sys._MEIPASS
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        
        dirname = os.path.join(application_path, subd)
             
        return open(dirname+filename).read()

    def Execute(self, sql):
        with pymssql.connect(server = self.server, database = 'ComEd_Cust') as conn:
        #with self.engine.connect() as conn:        
            conn.execute(sql)
            
    
    def ExecuteSQL(self, sql, **kwds):

        with pymssql.connect(server = self.server, database = self.db) as conn:

        #with self.engine.connect() as conn:                    
            return pd.read_sql(sql, conn, **kwds)
    
 
    # method to Import any data frame
    #@classmethod
    def ImportDataFrame(self, dataframe, table, schema, if_exists='fail', index=False, method='multi'):   
        #Conflict with fast importing data frames. Seems to work without index.
        with self.engine.connect() as conn: 
        #pandas to_sql doesn't play well with pymssql connection. Might need to make the engine in sqlalchemy
        #with pymssql.connect(server = self.server, database = self.db) as conn:

            if (index == True and method =='multi'):  
                #SQL server only accepts 2100 parameters per insert
                new = dataframe.reset_index()        
                chunknum = math.floor(2100/len(new.columns))  -1            
                new.to_sql(table, conn, schema, if_exists=if_exists, index=False,chunksize=chunknum, method=method   )    
            else:
                dataframe.to_sql(table, conn, schema, if_exists=if_exists, index=index  )   


## Interface For a SQLite Native DB
# It is sometimes useful to download and store data in order for a faster development cycle (less waiting on data)
class NativeDB(DBConnection):
    def __init__(self,db, server='',readonly=False):
        
        self.engine = s.create_engine("sqlite:///%s" % db, execution_options={"sqlite_raw_colnames": True})
        #self.local_connection = self.local_engine.connect()
        

    def GetLabeledData(self, LabelKeys):        
        LabelKeys['start'] = LabelKeys['start'].astype(np.int64)     
        LabelKeys['end'] = LabelKeys['end'].astype(np.int64)                        
        return self.ExecuteSQL("Select * from device_year_week_labels")        
    
    def GetIntervalData(self, devices, start, end, channels=['consumption','generation','voltage']):
        
        device_str = ','.join(str(device) for device in devices)        
        channel_str = ','.join(str(channel) for channel in channels)

        unix_start = pd.to_datetime(start).value
        unix_end = pd.to_datetime(end).value                

        template = 'Select deviceid, valuedate, {channels} from IntervalData where deviceid in ({devices}) and valuedate >= {start} and valuedate <= {end}'

        sql = template.format(devices=device_str,channels=channel_str, start=unix_start, end=unix_end)      
        df = self.ExecuteSQL(sql)
        df['valuedate'] = pd.to_datetime(df['valuedate'])
        df = df.set_index(['deviceid','valuedate'])
        
        return  df      

    def ImportIntervalData(self, dataframe):
        tempname = 'stage_IntervalData'   
        dataframe.reset_index(inplace=True)
        dataframe['valuedate'] = dataframe['valuedate'].astype(np.int64)     
        
        self.ImportDataFrame(dataframe, tempname,schema='main', if_exists='replace')        
        self.Execute('Insert or Replace into IntervalData Select valuedate, deviceid, consumption, generation, voltage from {};'.format(tempname))

        return


    #list table in local cache
    def ListTables(self):
        return self.ExecuteSQL("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")


    def UpdateLabels(self, deviceid, year, week, labels):

        sql = """INSERT OR REPLACE INTO 
        device_year_week_labels(deviceid, year, week, labels) VALUES({deviceid},{year},{week},'{labels}');
        """.format(deviceid=deviceid, year=year, week=week, labels=labels)
        return self.Execute(sql)


    def GetLabels(self, deviceid, year, week):
        
        sql = """Select labels from device_year_week_labels where deviceid = {deviceid} and year={year}
        and week={week};
        """.format(deviceid=deviceid, year=year, week=week)
        return self.ExecuteSQL(sql)



## Data Helper Interfacing with OO
class OODataHelper(DBConnection):
    def __init__(self, server, db): 
        super(OODataHelper, self).__init__(server, db)

        self.valuevars = ["Value"+"{0:0=2d}".format(i) for i in range(1,25)]       
        self.engine = s.create_engine('mssql+pyodbc://{server}/{db}?trusted_connection=True&driver=ODBC+Driver+17+for+SQL+Server'.format(server=server, db=db),pool_recycle=3600 )                

        if db != 'ETL':
            self.ChannelDict = self.OO_GetChannelIDs()

            self.IntervalSQLTemplate = self.LoadSQLFile('IntervalDatabyDevice.sql')     

        return


    def GetAMIDataStats(self):
        return self.ExecuteSQL('select * from dbo.ProjectAMIDataStats')

    def GetChannelStorageIds(self, column, values):
        vals = "','".join(values)                
        sql = self.LoadSQLFile('GetChannelStorageIDs.sql').format(column=column, values=vals) 

        return self.ExecuteSQL(sql)['channelid'].values

    #Get channelids based on settings
    def OO_GetChannelIDs(self, type='interval'):
        
        sql = self.LoadSQLFile('GetChannelIdsFromSettings.sql')#.format(type=type)
        df = self.ExecuteSQL(sql)
        df['ChannelID']=df['Value'].str.split(',')
        df = df.explode('ChannelID').reset_index() 
        df['ChannelID']= df['ChannelID'].astype(int)
        #mask = df['ChannelName'].isin(channels)
        df = df[['ChannelName','ChannelID']]#[mask]    
        return df.set_index('ChannelID').to_dict(orient='dict')['ChannelName']   

    def UnpivotChannelValueWide(self, df):

        df['ValueDate']= pd.to_datetime(df['ValueDate'])          
        df['ChannelName'] = df['ChannelID'].map(self.ChannelDict)
        m = df.melt(value_vars=self.valuevars, id_vars=['DeviceID','ValueDate','ChannelName','TimeOffSetInMinutes'])
        
        #Create Timestamps from date, hour and timeoffset pieces
        m['TimeOffSetInMinutes'] = pd.to_timedelta(m['TimeOffSetInMinutes'].astype(int), unit='minute')
        m['hour']= pd.to_timedelta(m['variable'].str[5:].astype(int) - 1, unit='hour')
        m['ValueDate']=(pd.to_datetime(m['ValueDate']) + m['hour'] + m['TimeOffSetInMinutes'])                
        #Map channelids to common channel names
        dff = pd.pivot_table(data=m[['DeviceID', 'ValueDate','ChannelName','value']],index=['DeviceID','ValueDate'],values=['value'], columns=['ChannelName'])['value'].reset_index()
        dff['ValueDate'] = pd.to_datetime(dff['ValueDate'])
        dff = dff.set_index(['DeviceID','ValueDate'])
        return dff

    def GetIntervalData(self, devices, start, end,channels = ['consumption', 'generation','voltage']):                
                
        #Create , seperated string of devices and channelids
        d = ','.join([str(device) for device in devices])
        c = ','.join([str(key) for key, value in self.ChannelDict.items() if value in channels])
        
        #Format sql with strings above and get raw data             
        sql=''
        template = self.IntervalSQLTemplate
        if isinstance(start, list):
            template = template.split('--')
            data_sql = template[0].format(channels=c)
            device_sql_tamplate = template[1]
            temp = []
            for i in range(len(devices)):
                temp.append(device_sql_tamplate.format(devices=devices[i], start=start[i], end=end[i]))
            device_sql = ') or ('.join(temp)
            sql = data_sql + '('+device_sql+')'
        else:
            sql = self.IntervalSQLTemplate.format(devices=d, channels=c, start=start, end=end)
        
        df = self.ExecuteSQL(sql) 

        if df.empty == False:
            return self.UnpivotChannelValueWide(df)
        else:
            return df

    #Load a Query Builder View by name on a customer
    def LoadQueryBuilderView(self, name):
        sql = """SELECT [SQL] as query_text
        FROM [ComEd_Cust].[dbo].[ViewLookupQueryBuilderView]
        where name = '{QBV}'""".format(QBV=name)
    
        qbv_sql = self.ExecuteSQL(sql)['query_text'].values[0]

        return self.ExecuteSQL(qbv_sql)


#Class for accessing weather data.
class WeatherDB(DBConnection):

    def __init__(self,server, db): 
                   
        self.engine = ""
        """s.create_engine('mssql+pyodbc://{server}/{db}?trusted_connection=True&driver=ODBC+Driver+13+for+SQL+Server'.format(server=server, db=db),pool_recycle=3600, fast_executemany=True,
        connect_args = {'connect_timeout': 60})"""        
        super(WeatherDB, self).__init__(server, db)
    
        return

    #Returns hourly temperature by zip code and date range [start, end (up to midnight)]
    def GetWeather(self, zips, start, end):
        
        zips = "','".join([str(zip) for zip in zips])

        sql = """Select ZipCode as Zip,
        Dateadd(hour, [Hour],Cast([Date] as smalldatetime)) as ValueDate,
        Temp
        from dbo.CorrectedHourly c
        inner join dbo.ZipCode z on z.zipcodeid = c.zipcodeid
        where zipcode in ('{zips}') and Date >= '{start}' and Date <= '{end}'
        """.format(zips=zips, start=start, end=end)
        
        temp = self.ExecuteSQL(sql)

        return temp 

#Connection factory to get database connections from LV
#Need to connect to Azure databases in similar way via customer ID or DB name.
class ConnectionFactory():
    
    def __init__(self):
        
        self.helper = None 
    
    def GetConnection(self, ClientID, UseStaging):
        if str(ClientID) == 'local':            
            return NativeDB(config.localdb)

        if str(ClientID) =='Weather':
            return WeatherDB(config.weather_serv, config.weather_db)
        elif int(ClientID) > 0:

            
            if UseStaging == True:
                self.helper = OODataHelper(server=config.stage_serv, db=config.stage_db)
            else:
                self.helper = OODataHelper(server=config.connection_serv, db=config.connection_db)

            sql = 'Select CustDBServerName, CustDBName from dbo.Customer where ClientID = {}'.format(ClientID)
            df = self.helper.ExecuteSQL(sql)                         
                     
            server, db = df['CustDBServerName'].values[0].split('.')[0], df['CustDBName'].values[0]                      

            #Return a new Database connection object
            #"TO DO convert to singleton object that uses one instance of the object and manages multiple connections'
            return OODataHelper(server, db)            

        else:
            print('Not a valid ClientID')

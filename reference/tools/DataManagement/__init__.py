

from DataManagement.DataHelpers import ConnectionFactory


conn_fac = ConnectionFactory()

def GetConnection(clientID, UseStaging=False):
    return conn_fac.GetConnection(clientID, UseStaging)
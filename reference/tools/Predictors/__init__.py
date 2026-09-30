#Goal is to provide accessibility to previously built predictors for use in pipelines or further research.

#This Module follows the factory pattern to instantiate and return predictor objects (generally wrapped around an sklearn or keras model) 

from Predictors.Predictors import *

pred_fac = PredictorFactory()

def GetPredictor(name, **kwds):
    return pred_fac.create_predictor(name, **kwds)
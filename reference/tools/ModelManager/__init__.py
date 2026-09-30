#This Module is used to provide access to saving and loading binary models.

import pickle
import os
import sys


subd = 'Models/'

# determine if application is a script file or frozen exe

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

dirname = os.path.join(application_path, subd)


#Load models 
def LoadModel(modelname, isNeuralNet=False):
    
    file = dirname+modelname
    if isNeuralNet == False:        
        return pickle.load(open(file+'.pickle', "rb"))
    else:    
        from tensorflow.keras.models import load_model #Don't load tensorflow until you have to?
        return load_model(file+'.h5')

def SaveModel(obj, modelname, isNeuralNet=False):
    file = dirname+modelname
    if isNeuralNet == False:        
        with open(file+'.pickle', "wb") as output_file:
            pickle.dump(obj, output_file)
    else:        
        from tensorflow.keras.models import load_model
        obj.save(file+'.h5') #save using neural net object itself


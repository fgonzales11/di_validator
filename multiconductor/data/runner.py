import multiconductor as mc
from multiconductor.pycci import app

filename='data/MV_open_ring_pp.xlsx'
net = mc.from_excel(filename)

#%%
# Build the grid model (admittance matrices for passive elements)
net=app.build_model(net)
# Run multi-conductor power flow with current-injection method...
# LF=app.run_ts(Network)
net=app.run_snap(net)

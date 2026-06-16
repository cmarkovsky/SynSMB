# test_plot_decomposition.py
from syn_smb import SMBDataLoader, Preprocessor

smb = SMBDataLoader("./data/sectors_smb/Pine_Island_smb.nc").load()

pp = Preprocessor()
pp.fit(smb)
pp.plot_decomposition(smb)
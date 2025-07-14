import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

class Plotter:
    def __init__(self, smb: xr.DataArray, annual_smb: xr.DataArray, region: str = 'PIG'):
        self.smb = smb
        self.annual_smb = annual_smb
        self.region = region

    def plot_smb(self):
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplot_mosaic([['a)'], ['b)']], figsize = (12,8), sharex = False, sharey = False, layout = 'constrained')
            ax0 = axes['a)']
            ax1 = axes['b)']
            sns.lineplot(x=self.smb['time'], y=self.smb, ax=ax0, label='SMB')
            ax0.set_title(f'Surface Mass Balance (SMB) for {self.region}')
            ax0.set_xlabel('Time')
            ax0.set_ylabel('SMB (m w.e.)')

            sns.lineplot(x=self.annual_smb['year'], y=self.annual_smb, ax=ax1, label='Annual SMB')
            ax1.set_title(f'Annual Surface Mass Balance (SMB) for {self.region}')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Annual SMB (m w.e.)')
            

            plt.show()

    def plot_annual_smb(self):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.annual_smb['year'], y=self.annual_smb)
        plt.title('Annual Surface Mass Balance (SMB)')
        plt.xlabel('Year')
        plt.ylabel('Annual SMB (m w.e.)')
        plt.show()

    def plot_filtered_smb(self, filtered_smb: xr.DataArray, n_years: int):

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=self.smb['time'], y=self.smb, label='Original SMB')
        sns.lineplot(x=filtered_smb['time'], y=filtered_smb, label='Filtered SMB')
        plt.xlabel('Time')
        plt.ylabel('Normalized SMB')
        plt.legend()
        # plt.show()
    

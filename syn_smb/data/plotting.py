import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

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
        sns.lineplot(x=self.annual_smb['year'], y=self.annual_smb)
        plt.title('Annual Surface Mass Balance (SMB)')
        plt.xlabel('Year')
        plt.ylabel('Annual SMB (m w.e.)')
        plt.show()

    def plot_filtered_smb(self, filtered_smb: xr.DataArray, n_years: int, annual: bool = False):

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        if annual:
            sns.lineplot(x=self.annual_smb['year'], y=self.annual_smb, label='Original Annual SMB')
            sns.lineplot(x=filtered_smb['year'], y=filtered_smb, label=f'Filtered SMB - {n_years} Year(s)', color='tab:orange')
        else:
            sns.lineplot(x=self.smb['time'], y=self.smb, label='Original SMB')
            sns.lineplot(x=filtered_smb['time'], y=filtered_smb + self.smb.mean(dim='time'), label='Filtered SMB')
        plt.xlabel('Time')
        plt.ylabel('SMB (m w.e.)')
        plt.legend()
        # plt.show()
    
    def plot_pacf(self, data: xr.DataArray):
        """
        Plot the Partial Autocorrelation Function (PACF) of the data.
        """

        plot_pacf(data.values)
        plt.title(f'Partial Autocorrelation Function (PACF) for {data.name}')
        plt.xlabel('Lags')
        plt.ylabel('PACF')

    def plot_acf(self, data: xr.DataArray):
        """
        Plot the Autocorrelation Function (ACF) of the data.
        """

        plot_acf(data.values)
        plt.title(f'Autocorrelation Function (ACF) for {data.name}')
        plt.xlabel('Lags')
        plt.ylabel('ACF')

    def plot_filtered_smbs(self, filtered_smbs: dict):
        """
        Plot all filtered SMB data.
        """
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(nrows= len(filtered_smbs), figsize=(12, 8), sharex=True)

        for ax, (n_years, filtered_smb) in zip(axes, filtered_smbs.items()):
            sns.lineplot(x=self.smb['time'], y=self.smb, ax=ax, label='Original SMB')
            ax.set_title(f'Filtered SMB - {n_years} Year(s)')
            sns.lineplot(x=filtered_smb['time'], y=filtered_smb + self.smb.mean(dim='time'), ax=ax, label=f'Filtered SMB - {n_years} Year(s)')
            ax.set_xlabel('Time')
            ax.set_ylabel('SMB (m w.e.)')
            ax.set_title(f'')

        plt.tight_layout()
        plt.ylabel('SMB (m w.e.)')
        plt.legend()
        # plt.show()
    
    
    

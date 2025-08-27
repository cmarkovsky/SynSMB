from matplotlib import pyplot as plt
from syn_smb import SMBDataSet, Generator, Plotter, Generator2, SyntheticGenerator
from pandas.plotting import autocorrelation_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    reg = 'PIG'

    smb_ds = SMBDataSet(f"examples/data/era5_{reg}.grib")
    # print(f"SMB: {smb_ds.raw_tp.tp[-12:-1]}")
    # smb_ds.filter_smb(1)
    # smb_ds.filter_smb(10)
    # smb_ds.filter_smb(25)
    # smb_ds.plot_filtered_smbs()
    # plotter = Plotter(smb_ds.get_smb(), smb_ds.get_annual_smb(), region=reg)
    # plotter.plot_annual_smb()
    # generator = Generator2(smb_ds.get_smb())
    # generator.generate_smb(plot=True)
    gen = SyntheticGenerator(smb_ds.get_smb())
    gen.fit_ar_models()
    # betas = gen.check_psds()
if __name__ == "__main__":
    main()

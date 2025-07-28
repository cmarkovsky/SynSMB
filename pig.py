from matplotlib import pyplot as plt
from syn_smb import SMBDataSet, Generator
from pandas.plotting import autocorrelation_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    reg = 'PIG'

    smb_ds = SMBDataSet(f"examples/data/era5_{reg}.grib")

    generator = Generator(smb_ds.get_smb())
    generator.generate_smb(plot=True)

if __name__ == "__main__":
    main()

from syn_smb import SMBDataSet

def main():
    reg = 'PIG'

    smb_ds = SMBDataSet(f"examples/data/era5_{reg}.grib")

    smb_ds.forecast_smb(filt_center=1, n_years=10, plot=True)

    # preprocessor = Preprocessor(f"examples/data/era5_{reg}.grib")
    # t2m, tp = preprocessor.get_data()
    # print(f"Loaded datasets for {reg}:")

    # smb = preprocessor.get_smb()
    # smb_norm = preprocessor.get_normalized_smb()
    # print(f"Calculated SMB for {reg}: {smb}")
    # annual_smb = preprocessor.get_annual_smb()

    # plotter = Plotter(smb_norm, annual_smb)

    # bandpass = BandpassFilter(n_years=1)
    # filtered_smb = bandpass.filter(smb_norm)

    # plotter.plot_filtered_smb(filtered_smb)
    # print(f"Calculated annual SMB for {reg}: {annual_smb}")
    # smb_norm = preprocessor.get_normalized_smb()
    # print(f"Normalized SMB for {reg}: {smb_norm}")
    # preprocessor.normalize()
    # preprocessor.augment()
    # data = preprocessor.get_data()
    # print(data)

if __name__ == "__main__":
    main()

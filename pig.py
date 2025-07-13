from syn_smb import Preprocessor

def main():
    reg = 'PIG'
    preprocessor = Preprocessor(f"examples/data/era5_{reg}.grib")
    t2m, tp = preprocessor.get_data()
    print(f"Loaded datasets for {reg}:")

    smb = preprocessor.get_smb()
    print(f"Calculated SMB for {reg}: {smb}")
    # annual_smb = preprocessor.get_annual_smb()
    # print(f"Calculated annual SMB for {reg}: {annual_smb}") 
    # smb_norm = preprocessor.get_normalized_smb()
    # print(f"Normalized SMB for {reg}: {smb_norm}")
    # preprocessor.normalize()
    # preprocessor.augment()
    # data = preprocessor.get_data()
    # print(data)

if __name__ == "__main__":
    main()

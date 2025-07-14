from syn_smb import Preprocessor
import xarray as xr

def test_preprocess():
    # Create a sample dataset
    data_path = "examples/data/era5_PIG.grib"
    
    # Initialize the Preprocessor object
    preprocessor = Preprocessor(data_path)
    
    # Test if the data is correctly stored
    # assert preprocessor.get_data() == sample_data, "Preprocessor did not return the expected data"
    
    # Test if the data type is correct
    assert isinstance(preprocessor.get_smb(), xr.DataArray), "Data should be of type xarray.DataArray"

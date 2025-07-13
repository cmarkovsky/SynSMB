from syn_smb import Preprocessor

def test_preprocess():
    # Create a sample dataset
    sample_data = [1, 2, 3, 4, 5]
    
    # Initialize the Preprocessor object
    preprocessor = Preprocessor(sample_data)
    
    # Test if the data is correctly stored
    assert preprocessor.get_data() == sample_data, "Preprocessor did not return the expected data"
    
    # Test if the data type is correct
    assert isinstance(preprocessor.get_data(), list), "Data should be of type list"
    
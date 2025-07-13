from syn_smb.dataset import Dataset

def test_dataset():
    # Create a sample dataset
    sample_data = [1, 2, 3, 4, 5]
    
    # Initialize the Dataset object
    dataset = Dataset(sample_data)
    
    # Test if the data is correctly stored
    assert dataset.get_data() == sample_data, "Dataset did not return the expected data"
    
    # Test if the data type is correct
    assert isinstance(dataset.get_data(), list), "Data should be of type list"
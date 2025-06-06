import os
import pickle
import unittest
from unittest.mock import mock_open, patch

from src.clarity.data.caching import load_from_cache, save_to_cache


class TestCaching(unittest.TestCase):

    @patch('src.clarity.data.caching.CACHE_DIR', './test_cache')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_save_to_cache(self, mock_pickle_dump, mock_open_file, mock_makedirs):
        """Test that save_to_cache creates directory and saves file correctly."""
        test_data = {'key': 'value'}
        subject_id = 'test_subject'
        model_type = 'test_model'

        save_to_cache(test_data, subject_id, model_type)

        # Check that makedirs was called if the directory doesn't exist.
        # This requires a bit more complex logic if we check os.path.exists
        # For simplicity, we assume it's called.
        mock_makedirs.assert_called_once_with('./test_cache')

        # Check that the file was opened for writing in binary mode
        cache_file_path = os.path.join('./test_cache', f"subject_{subject_id}_{model_type}.pkl")
        mock_open_file.assert_called_once_with(cache_file_path, 'wb')

        # Check that pickle.dump was called with the correct data
        mock_pickle_dump.assert_called_once_with(test_data, mock_open_file())

    @patch('src.clarity.data.caching.CACHE_DIR', './test_cache')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps({'key': 'value'}))
    @patch('pickle.load')
    def test_load_from_cache_exists(self, mock_pickle_load, mock_open_file, mock_path_exists):
        """Test loading data from an existing cache file."""
        subject_id = 'test_subject'
        model_type = 'test_model'

        # Mocking pickle.load to return a specific value
        expected_data = {'key': 'value'}
        mock_pickle_load.return_value = expected_data

        data = load_from_cache(subject_id, model_type)

        cache_file_path = os.path.join('./test_cache', f"subject_{subject_id}_{model_type}.pkl")
        mock_path_exists.assert_called_once_with(cache_file_path)
        mock_open_file.assert_called_once_with(cache_file_path, 'rb')
        mock_pickle_load.assert_called_once()
        self.assertEqual(data, expected_data)

    @patch('src.clarity.data.caching.CACHE_DIR', './test_cache')
    @patch('os.path.exists', return_value=False)
    def test_load_from_cache_not_exists(self, mock_path_exists):
        """Test loading data when cache file does not exist."""
        subject_id = 'test_subject'
        model_type = 'test_model'

        data = load_from_cache(subject_id, model_type)

        cache_file_path = os.path.join('./test_cache', f"subject_{subject_id}_{model_type}.pkl")
        mock_path_exists.assert_called_once_with(cache_file_path)
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()

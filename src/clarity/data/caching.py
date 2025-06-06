import os
import pickle
from ..training.config import CACHE_DIR

def save_to_cache(data, subject_id, model_type):
    """Saves processed data for a subject to a cache file."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    cache_file = os.path.join(CACHE_DIR, f"subject_{subject_id}_{model_type}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved processed data for subject {subject_id} to cache.")

def load_from_cache(subject_id, model_type):
    """Loads processed data for a subject from a cache file if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"subject_{subject_id}_{model_type}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            print(f"Loading processed data for subject {subject_id} from cache.")
            return pickle.load(f)
    return None 
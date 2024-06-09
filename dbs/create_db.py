import os
import random
import sqlite3
import sys
import yaml
from pathlib import Path

# Ensure the parent directory is in the sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from db_model import AudioDatabase, gather_wav_files

# Define the relative path to the constants.yaml file
current_dir = Path(__file__).resolve().parent
yaml_file_path = current_dir / '../constants.yaml'

# Load constants from YAML file
with open(yaml_file_path, "r") as file:
    config = yaml.safe_load(file)

# Extracting constants from the YAML config
root_data_folder = config['root_data_folder']
db_path_train = "dbs//audio_files_Train.db"
db_path_test = "dbs//audio_files_Test.db"

# Create database instances
audio_db_train = AudioDatabase(db_path_train)
audio_db_test = AudioDatabase(db_path_test)

# Gather and insert train files into the database
train_files = gather_wav_files(root_data_folder, "Train", all_files=1)
audio_db_train.insert_files(train_files)

# Gather and insert test files into the database
test_files = gather_wav_files(root_data_folder, "Test", all_files=1)
audio_db_test.insert_files(test_files)



print("randfile 1")
print(audio_db_test.get_random_file(audio_db_test.get_random_speaker()))

print("randfile 2")
print(audio_db_test.get_random_file(audio_db_test.get_random_speaker()))

print("randfile 3")
print(audio_db_test.get_random_file())

print("randfile 4")
print(audio_db_test.get_random_file())

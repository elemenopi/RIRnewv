import os
import random
import sqlite3
class AudioDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS speaker_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    file_path TEXT NOT NULL
                )
            """)
    def count_distinct_speakers(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT speaker) FROM speaker_files")
        result = cursor.fetchone()
        return result[0] if result else 0

    def insert_files(self, speaker_files):
        with self.conn:
            for speaker, files in speaker_files.items():
                if isinstance(files, list):
                    for file in files:
                        self.conn.execute("INSERT INTO speaker_files (speaker, file_path) VALUES (?, ?)", (speaker, file))
                else:
                    self.conn.execute("INSERT INTO speaker_files (speaker, file_path) VALUES (?, ?)", (speaker, files))

    def get_random_file(self, speaker):
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path FROM speaker_files WHERE speaker = ? ORDER BY RANDOM() LIMIT 1", (speaker,))
        result = cursor.fetchone()
        return result[0].replace("\\", "*").replace("/", "*").replace("*","\\") if result else None
    
    def get_random_speaker(self):
        cursor = self.conn.execute("SELECT DISTINCT speaker FROM speaker_files")
        speakers = cursor.fetchall()
        if speakers:
            return random.choice(speakers)[0]
        else:
            return None

    def get_all_data(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM speaker_files")
        return cursor.fetchall()

def gather_wav_files(root_folder: str, train_test: str, all_files: int) -> dict:
    in_folder = os.path.join(root_folder, train_test)
    if not os.path.exists(in_folder):
        print(f"{train_test} folder not found.")
        return {}

    output_folders = {}  # speaker: [sound_files...]

    for speaker_folder in os.listdir(in_folder):
        speaker_path = os.path.join(in_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            sound_files = []
            for sound_folder in os.listdir(speaker_path):
                sound_path = os.path.join(speaker_path, sound_folder)
                if os.path.isdir(sound_path):
                    for filename in os.listdir(sound_path):
                        if filename.endswith(".wav"):
                            src_filepath = os.path.join(sound_path, filename)
                            sound_files.append(src_filepath)
            if all_files == 0 and sound_files:
                output_folders[speaker_folder] = random.choice(sound_files)
            else:
                output_folders[speaker_folder] = sound_files

    return output_folders

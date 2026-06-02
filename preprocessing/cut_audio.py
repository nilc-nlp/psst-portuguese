import pandas as pd
import subprocess
import os

BASE_DIR = "catna-audios"
OUTPUT_DIR = "catna-audios-clean-cut"

def cut_string_until_period(text):
    if '.' in text:
        return text.split('.')[0]
    return text

def cut_audio_ffmpeg(input_path, start, end, output_path):
    duration = end - start

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",                  # overwrite
        "-ss", str(start),     # start (seconds)
        "-i", input_path,
        "-t", str(duration),   # duration (seconds)
        "-c", "copy",          # no re-encode (fast)
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


df = pd.read_csv("catna_clean.csv")
new_df = df.copy()

current_file = None
current_file_index = 0
file_name_list = []

for index, row in df.iterrows():
    if cut_string_until_period(row['File']) != current_file:
        print("New file detected.")
        current_file = cut_string_until_period(row['File'])
        current_file_index = 0

    if index % 100 == 0:
        print(f"Processing row {index}...")
        print(f"Current file: {current_file}, index: {current_file_index}")

    file_path = f"{BASE_DIR}/{current_file}.wav"
    start_time = float(row['Start'])
    end_time = float(row['End'])

    output_path = f"{OUTPUT_DIR}/{current_file}/{current_file}_{current_file_index}.wav"
    file_name_list.append(output_path)

    cut_audio_ffmpeg(file_path, start_time, end_time, output_path)

    current_file_index += 1


new_df['File_Path'] = file_name_list
new_df = new_df.loc[:, [
    'File_Path', 'File', 'Speaker',
    'Number_Concatenated', 'Index_Concatenated',
    'Start', 'End', 'Duration', 'Text'
]]

new_df.to_csv("catna_clean_with_paths.csv", index=False)
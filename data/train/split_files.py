import os
import shutil

# Path to the 'train' folder
#train_folder = 'train_split'

# Create a new directory to store the grouped files
#grouped_folder = 'train_split'
#os.makedirs(grouped_folder, exist_ok=True)

# Get a list of all files in the 'train' folder
#files = os.listdir(train_folder)

# Number of files per group
#files_per_group = 100

# Split the files into groups and move them to the new directory
'''for i in range(0, len(files), files_per_group):
    group_files = files[i:i + files_per_group]
    group_folder = os.path.join(grouped_folder, f'group_{i // files_per_group + 1}')
    os.makedirs(group_folder, exist_ok=True)
    for file in group_files:
        shutil.move(os.path.join(train_folder, file), os.path.join(group_folder, file))

print(f"Files have been grouped into folders with less than {files_per_group} files each.")'''


# Path to the 'grouped_train' folder
grouped_folder = 'train_split'

# Get a list of all subfolders in the 'grouped_train' folder
subfolders = [f.path for f in os.scandir(grouped_folder) if f.is_dir()]

# Count the number of .wav files in each subfolder
for subfolder in subfolders:
    wav_files = [file for file in os.listdir(subfolder) if file.endswith('.wav')]
    print(f"{subfolder} contains {len(wav_files)} .wav files.")
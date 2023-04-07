import os
import shutil

# Uses the List_of_testing_videos.txt from the Celeb-DF-v2 dataset to move the testing files into a separate directory
# based on the label

# Path to the file containing the list of files to move
file_list_path = '../datasets/Celeb-DF-v2/List_of_testing_videos.txt'

# Path to the directory where files with 0/1 will be moved
zero_dir = "../datasets/Celeb-DF-v2/testing-0/"
one_dir = "../datasets/Celeb-DF-v2/testing-1/"

# Path to the base Celeb-DF-v2
celeb_df2_dir = "../datasets/Celeb-DF-v2/"

# Open the file containing the list of files
with open(file_list_path) as f:
    for line in f:
        # Split the line into the label and the file path
        label, local_file_path = line.strip().split(' ')
        file_path = celeb_df2_dir + local_file_path

        # Move the file to the appropriate directory based on the label
        if label == '0':
            shutil.move(file_path, zero_dir)
        elif label == '1':
            shutil.move(file_path, one_dir)
        else:
            print(f'Invalid label {label} for file {file_path}')

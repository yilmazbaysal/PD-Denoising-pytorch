import os
from shutil import copyfile

for full_path, dir_names, file_names in os.walk("../../sidd/"):
    
    if 'our_set' not in full_path:
        for file_name in file_names:
            if 'GT' in file_name and '5_9' in file_name:
                copyfile(os.path.join(full_path, file_name), f"our_set/ground_truth/{full_path.split('/')[1]}_5_9.png")
            elif 'GT' in file_name and '5_10' in file_name:
                copyfile(os.path.join(full_path, file_name), f"our_set/ground_truth/{full_path.split('/')[1]}_5_10.png")
            elif 'NOISY' in file_name and '5_9' in file_name:
                copyfile(os.path.join(full_path, file_name), f"our_set/noisy/{full_path.split('/')[1]}_5_9.png")
            elif 'NOISY' in file_name and '5_10' in file_name:
                copyfile(os.path.join(full_path, file_name), f"our_set/noisy/{full_path.split('/')[1]}_5_10.png")

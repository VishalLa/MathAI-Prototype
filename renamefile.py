import os 

def rename_file_in_folder(folder_path, prefix):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    for index, filename in enumerate(os.listdir(folder_path)):
        old_file_path = os.path.join(folder_path, filename)

        if os.path.isdir(old_file_path):
            continue

        file_extension = os.path.splitext(filename)[1]

        new_filename = f"{prefix}-{index+1}{file_extension}"
        new_file_path = os.path.join(folder_path, new_filename)

        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")


folder_path = r'C:\\Users\\visha\\OneDrive\\Desktop\\New folder\\Model\\entiredataset\\symbols\\x'
prefix = 'x'
rename_file_in_folder(folder_path, prefix)


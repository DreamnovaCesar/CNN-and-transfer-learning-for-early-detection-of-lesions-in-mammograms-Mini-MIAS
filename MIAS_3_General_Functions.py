import os

# Sort Files

def ShowSort(Folder_Path): 

	"""
	Read all images in a folder and sort them.

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	int:Returning value
    int:Returning list[str]

   	"""

	NumberImages = len(os.listdir(Folder_Path))

	print("\n")
	print("********************************")
	print(f"Images: {NumberImages}")
	print("********************************")
	print("\n")

	files = os.listdir(Folder_Path)
	print(files)
	print("\n")

	print("********************************")
	sorted_files =  sorted(files)
	print(sorted_files)
	print("\n")
	print("********************************")

	return sorted_files, NumberImages

# Remove all files in folder

def Removeallfiles(Folder_Path):

	"""
	Remove all images inside the folder chosen

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	Void

   	"""

	for File in os.listdir(Folder_Path):
		filename, extension  = os.path.splitext(File)
		print(f"Removing {filename} ✅")
		os.remove(os.path.join(Folder_Path, File))

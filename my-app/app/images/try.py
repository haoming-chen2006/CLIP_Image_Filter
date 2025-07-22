file_path = "/pscratch/sd/h/haoming/Projects/clip/image_paths.csv"
with open(file_path, 'r') as file:
    image_paths = [line.strip() for line in file if line.strip()]
print(image_paths)
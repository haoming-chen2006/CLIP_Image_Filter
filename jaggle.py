import kagglehub

# Download latest version
path = kagglehub.dataset_download("alexanderliao/artbench10")

print("Path to dataset files:", path)
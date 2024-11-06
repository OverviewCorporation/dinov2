#!/bin/bash

# Specify the directory containing the files
directory="."

# Loop over all .trt files in the specified directory
for file in "$directory"/*.trt; do
  # Check if the file exists (in case there are no .trt files)
  if [[ -f "$file" ]]; then
    # Get the base name of the file (without the directory path)
    base_name=$(basename "$file")
    # Rename the file by prefixing '80i'
    cp "$file" "$directory/80i$base_name"
  fi
done

echo "Files renamed successfully."

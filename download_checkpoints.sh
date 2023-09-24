#!/bin/bash

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "Error: wget is not installed. Please install wget and try again."
    exit 1
fi

# Directory to store the downloaded files
DIR="checkpoints"

# Create the directory if it doesn't exist
mkdir -p "$DIR"

# URLs of the files to be downloaded
URL1="https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/mask3d.ckpt"
URL2="https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/human3d.ckpt"

# Download the files using wget
wget -P "$DIR" "$URL1"
wget -P "$DIR" "$URL2"

# Print a success message if both files are downloaded successfully
if [ $? -eq 0 ]; then
    echo "Files downloaded successfully to $DIR/"
else
    echo "There was an error downloading the files."
fi

import os
import csv

# Function to collect images and create dataset
def create_dataset(dataset_directory):
    # Open or create a CSV file to store image paths and labels
    with open('dataset.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'label'])

        # Traverse the dataset directory to collect images and assign labels
        for label_name in os.listdir(dataset_directory):
            label_directory = os.path.join(dataset_directory, label_name)
            if os.path.isdir(label_directory):
                for image_name in os.listdir(label_directory):
                    image_path = os.path.join(label_directory, image_name)
                    # Write image path and label to the CSV file
                    csv_writer.writerow([image_path, label_name])

def start():
    # Directory to save the dataset
    dataset_directory = 'dataset'

    # Create the directory if it doesn't exist
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    # Collect images and create dataset
    create_dataset(dataset_directory)

    # Your other code here...



import os
import csv


# appending all jpg or jpeg files path's to csv file
# csv file format - <path, label>
def format_paths_into_csv_name_label(root_dir, output_csv, label):
    # root_dir - folder which contains images that will be added to csv_paths
    # output_csv - "path/name.csv" csv file in which all images will be storing
    # label - all images labels (from 0 to 9)

    with open(output_csv, mode='a', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # iterating through all dirs/files
        for file in os.listdir(root_dir):
            # checking if file have right extension
            try:
                if file.endswith(".jpg") or file.endswith(".jpeg"):
                    writer.writerow([f"{root_dir}/{file}", label])
            except Exception as e:
                print(e)

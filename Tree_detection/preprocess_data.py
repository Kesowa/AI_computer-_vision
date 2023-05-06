import os
import glob
import re
import numpy as np
import pandas as pd
from deepforest import preprocess, utilities

# Set the file permission to readable by all users
file_path = r"C:\Users\FS-AI\Downloads\Tree canopy\Chayan_mandal"
os.chmod(file_path, 0o644)

# Read xml and create pandas frame
annotation_paths = glob.glob(file_path + "/*.xml")
# print(annotation_paths)
annotation_list = []
for xml in annotation_paths:
    xml_parse = utilities.xml_to_annotations(xml)

    # Append converted dataframe to file. Saved alongside the images. This is a temporary file for cutting windows.
    with open("FG_example.csv", "a") as f:
        xml_parse.to_csv(f, header=(f.tell()==0), index=False)

    # Get plot name to match to RGB image
    print("xml:", xml)
    plot_name_match = re.search("tree_canopy(\w+).xml", xml)
    if plot_name_match:
        plot_name = plot_name_match.group(1)
    else:
        print("No match found for pattern in xml:", xml)
        continue

    # Where to save cropped images
    crop_dir = "crops/"
    try:
        result = preprocess.split_raster(path_to_raster=f"{file_path}/tree_canopy{plot_name}.JPG",
                                         annotations_file="FG_example.csv",
                                         base_dir=crop_dir,
                                         patch_size=700,
                                         patch_overlap=0.05)
        annotation_list.append(result)
    except ValueError as e:
        print(f"Error processing {file_path}/tree_canopy{plot_name}.JPG: {e}")
    

train_annotations = pd.concat(annotation_list)

# Split image crops into training and test. Normally these would be different tiles! Just as an example.
image_paths = train_annotations.image_path.unique()
test_paths = np.random.choice(image_paths, 3)
test_annotations = train_annotations.loc[train_annotations.image_path.isin(test_paths)]
train_annotations = train_annotations.loc[~train_annotations.image_path.isin(test_paths)]

# View output
train_annotations.head()
print("There are {} training crown annotations".format(train_annotations.shape[0]))
print("There are {} test crown annotations".format(test_annotations.shape[0]))

# Save to file
# Write window annotations file without a header row, same location as the "base_dir" above.
train_annotations.to_csv(crop_dir + "train.csv", index=False, header=False)
test_annotations.to_csv(crop_dir + "test.csv", index=False, header=False)

annotations_file = crop_dir + "train.csv"
test_file = crop_dir + "test.csv"


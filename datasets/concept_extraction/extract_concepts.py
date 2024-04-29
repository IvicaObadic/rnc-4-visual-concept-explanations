import numpy as np
import os
from dataloader import labeled_data
from torch.utils.data import Dataset, DataLoader
import torch
import tifffile as tiff


def copy_rgb_data(save_path):
    # load 4 channel tiff and store as RGB png
    save_path = save_path
    img_names = []

    with open('img_names.txt', 'r') as f:
        lines = f.read().splitlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace('MSK', 'IMG')
        lines[i] = lines[i].replace('msk', 'img')
        lines[i] = lines[i].replace('labels', 'aerial')

    data_set = labeled_data(lines, transform='RGB')
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

    batch_size = 200

    for i, (path, data) in enumerate(data_loader):
        np_image = np.asarray(data)
        name_save = save_path+'/'+path[0].split('/')[-1]
        tiff.imsave(name_save, np_image)


# Set up file locations
root_path = 'flair_1_labels_train'  # alternatively use flair_1_labels_test
txt_save_name = 'img_names.txt'  # txt file with saved paths
im_save_path = 'rgb_data'

# Set goal indices (may be lists)
goal_idx = [[3, 1], [2]]

# Set upper and lower limit per goal idx
percentage_settings = [[0.1, 1], [0.5, 1]]


# Set up dataloader
img_names = []

for path, subdirs, files in os.walk(root_path):
    for name in files:
        img_names.append(os.path.join(path, name))

flair_data = labeled_data(img_names)
batch_size = 200

flair_dataloader = DataLoader(
    flair_data, batch_size=batch_size, shuffle=False, num_workers=4)


# init list
save_paths_idx = []

# iterate through data and check if conditions are fulfilled
for batch_idx, (path, data) in enumerate(flair_dataloader):
    choice_values = torch.ones([data.shape[0]])

    for i, (idx, percentages) in enumerate(zip(goal_idx, percentage_settings)):
        perc_goal = torch.zeros([data.shape[0]])

        for j in idx:
            perc_goal = perc_goal + torch.sum(data == j, dim=[1, 2])/(512*512)
        choice_values = choice_values * \
            (perc_goal >= percentages[0]) * (perc_goal <= percentages[1])
    idx_goal = torch.nonzero(choice_values).reshape(-1)
    list_idx = idx_goal.tolist()
    for i in list_idx:
        save_paths_idx.append(path[i])


# Print result
print(
    f"The Number of images fulfilling condition of goals is {len(save_paths_idx)}")


with open(txt_save_name, 'w') as f:
    for line in save_paths_idx:
        f.write(f"{line}\n")

print('Do you want to copy? y/n')
ans_copy = input()

# Write out data in
if ans_copy == 'y':
    copy_rgb_data(im_save_path)

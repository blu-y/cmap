import robotathome as rh
print(rh.__version__)
import pandas as pd
import os

from robotathome import RobotAtHome
from robotathome import logger, log, set_log_level
from robotathome import time_win2unixepoch, time_unixepoch2win
from robotathome import get_labeled_img, plot_labeled_img

log.set_log_level('INFO')  # SUCCESS is the default
level_no, level_name = log.get_current_log_level()
print(f'Current log level name: {level_name}')

my_rh_path = './dataset'
my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
my_scene_path = os.path.join(my_rh_path, 'files/scene')
my_wspc_path = './dataset/result'

try: 
      rh = RobotAtHome(my_rh_path, my_rgbd_path, my_scene_path, my_wspc_path)
except:
      logger.error("Something was wrong")

# The full dataset is returned by default
lblrgbd = rh.get_sensor_observations('lblrgbd')
print(f"# Labeled RGBD set: {len(lblrgbd)} observations with {len(lblrgbd.columns)} fields")
print(lblrgbd.info())

# Let's look any observation, e.g. id = 100000
id = 100000
[rgb_f, d_f] = rh.get_RGBD_files(id)
logger.info("Sensor observation {} files\n RGB file   : {}\n Depth file : {}", id, rgb_f, d_f )

from IPython.display import Image
Image(rgb_f)
Image(d_f)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Default (inches): rcParams["figure.figsize"] = [6.4, 4.8]
plt.rcParams['figure.figsize'] = [9.6, 7.2]

rgb_img = mpimg.imread(rgb_f)
# rgb_img is a numpy array
print(type(rgb_img))

# Another way to visualize it
imgplot = plt.imshow(rgb_img)

# Let's follow with the previous example
id = 100000
# Notice the masks parameter indicating that we are not interested
# in the corresponding label masks at this time, resulting in a faster
# response
labels = rh.get_RGBD_labels(id, masks = False)
# labels is a dataframe with the object annotations for the
# RGBD observation
logger.info("\nlabels: \n{}", labels)

# For example, let's get the category name of the fourth label
label_category_name = rh.id2name(labels.loc[3,'object_type_id'], 'ot')
print(f"4th label's object type/category: {label_category_name}")

# In this case we'll get a dataframe similar to the previous one
# but with an added mask column
labels_with_masks = rh.get_RGBD_labels(id)
# labels is a dataframe with the object annotations for the RGBD observation
logger.info("\nlabels with masks: \n{}", labels_with_masks)

# Again, we choose the previous label, i.e. the toilet
label_binary_mask = labels_with_masks.loc[3,'mask']
plt.imshow(label_binary_mask, cmap='gray')

# The following function returns an image patched with labels which
# can be easily plotted
[labeled_img, _] = get_labeled_img(labels_with_masks, rgb_f)
plt.imshow(labeled_img)
plt.show()

plot_labeled_img(labels_with_masks, rgb_f)

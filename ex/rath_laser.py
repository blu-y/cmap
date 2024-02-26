import robotathome as rh
print(rh.__version__)
import pandas as pd
import os

from robotathome import RobotAtHome
from robotathome import logger, log, set_log_level
from robotathome import time_win2unixepoch, time_unixepoch2win
from robotathome import get_labeled_img, plot_labeled_img, plot_scan, get_scan_xy

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
laser_obs = rh.get_sensor_observations('lsrscan')
logger.info(f"# Laser Scans set: {len(laser_obs)} observations with {len(laser_obs.columns)} fields")

laser_obs.info()

# For example, let's examine the first laser observation
id = 200000
laser_scan = rh.get_laser_scan(id)
laser_scan.info()

# To get a list with the shot values
# We can easily convert a column to a list
shot_list = laser_scan['scan'].values.tolist()
logger.info("\nShots column: \n{}\nShot list: \n{}...", laser_scan['scan'], shot_list[:5])

logger.info("\naperture    : {} radians \nmax_range   : {} meters\nno_of_shots : {}",
            laser_scan.aperture, laser_scan.max_range, laser_scan.no_of_shots )

xy = get_scan_xy(laser_scan)
logger.info("\n (x,y) laser scan coordinates: \n{}", xy)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9.6, 7.2]
plot_scan(laser_scan)


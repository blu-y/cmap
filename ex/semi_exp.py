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


"""Logging module for the project.
"""

import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "data/torchmanager/logs")
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Create the log directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] \n\tModule:: %(module)s \n\t\tLine:: %(lineno)s \n" \
        + "\t\t\tLevel::%(levelname)s \n\t\t\t\t%(message)s",
    # format="[ %(asctime)s ] %(lineno)d % (name)s - % (levelname)s - %(message)s",
)

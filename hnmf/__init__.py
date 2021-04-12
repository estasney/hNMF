from hnmf.model import *
from hnmf.helpers import *

import logging

logger = logging.getLogger("hnmf")
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())

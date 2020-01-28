from hNMF.model import *
from hNMF.helpers import *

import logging

logger = logging.getLogger('hNMF')
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())

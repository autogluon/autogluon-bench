import json
import time

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def formatted_time():
    current_time = time.localtime()
    formatted_time = time.strftime("%Y%m%dT%H%M%S", current_time)
    return formatted_time

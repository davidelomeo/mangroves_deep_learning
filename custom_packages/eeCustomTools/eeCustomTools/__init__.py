from .cloud_mask import * # noqa
from .compute_indices import * # noqa
from .image_segmentation import * # noqa
from .other_functions import * # noqa

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

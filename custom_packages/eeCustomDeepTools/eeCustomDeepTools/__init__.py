from .get_patches_info import * # noqa
from .records_split import * # noqa
from .fixed_length_features import * # noqa
from .prepare_batches import * # noqa
from .prepare_classes import * # noqa
from .prepare_predictions import * # noqa

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

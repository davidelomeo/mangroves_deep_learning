from .unet import * # noqa
from .vgg19_unet import * # noqa
from .resnet50_unet import * # noqa

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

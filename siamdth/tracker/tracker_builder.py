from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamdth.core.config import cfg
from siamdth.tracker.siamdah_tracker import SiamDTHTracker
TRACKS = {
          'SiamBANTracker': SiamDTHTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

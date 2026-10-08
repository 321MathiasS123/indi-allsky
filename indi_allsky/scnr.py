####################################################################################
# Algorithms below were borrowed from PixInsight docs
# https://www.pixinsight.com/doc/legacy/LE/21_noise_reduction/scnr/scnr.html
####################################################################################

#import time
#from pathlib import Path
#import time
import cv2
import numpy
import logging

from . import constants


logger = logging.getLogger('indi_allsky')


class IndiAllskyScnr(object):

    def __init__(self, config, night_av):
        self.config = config
        self.night_av = night_av

        self._night = None

        self.amount = self.config.get('SCNR_AMOUNT', 0.50)  # not currently used

        self._mtf_lut = None


    @property
    def night(self):
        return self._night

    @night.setter
    def night(self, new_night):
        self._night = int(bool(new_night))


    def additive_mask(self, scidata):
        ### The function below returns an out of memory error, needs to be fixed

        if len(scidata.shape) == 2:
            # grayscale
            return scidata

        #logger.warning('Applying SCNR additive mask')


        image_height, image_width = scidata.shape[:2]
        b, g, r = cv2.split(scidata)

        #start = time.time()

        ones = numpy.ones([image_height, image_width])

        m = numpy.minimum(ones, numpy.sum(r + b))

        #g * (1 - self.amount) * (1 - m) + m * g
        g = numpy.multiply(g * (1 - self.amount), numpy.sum(numpy.subtract(ones, m), numpy.multiply(m, g)))  # oom

        #elapsed_s = time.time() - start
        #logger.info('SCNR additive mask in %0.4f s', elapsed_s)

        return cv2.merge((b, g, r))


    def average_neutral(self, scidata):
        if len(scidata.shape) == 2:
            # grayscale
            return scidata

        #logger.warning('Applying SCNR average neutral')


        b, g, r = cv2.split(scidata)

        #start = time.time()

        # Widen the sum so neither 8-bit nor 16-bit input can overflow.
        sum_dtype = numpy.uint32 if scidata.dtype == numpy.uint16 else numpy.uint16
        m = numpy.add(r.astype(sum_dtype), b.astype(sum_dtype)) * 0.5
        g = numpy.minimum(g, m.astype(scidata.dtype))

        #elapsed_s = time.time() - start
        #logger.info('SCNR average neutral in %0.4f s', elapsed_s)

        return cv2.merge((b, g, r))


    def maximum_neutral(self, scidata):
        if len(scidata.shape) == 2:
            # grayscale
            return scidata

        #logger.warning('Applying SCNR maximum neutral')


        b, g, r = cv2.split(scidata)

        #start = time.time()

        m = numpy.maximum(r, b)
        g = numpy.minimum(g, m)

        #elapsed_s = time.time() - start
        #logger.info('SCNR maximum neutral in %0.4f s', elapsed_s)

        return cv2.merge((b, g, r))


    def green_mtf(self, scidata):
        if len(scidata.shape) == 2:
            # grayscale
            return scidata


        #mtf_start = time.time()


        if self.night != self.night_av[constants.NIGHT_NIGHT] or (
            self._mtf_lut is not None and self._mtf_lut.dtype != scidata.dtype
        ):
            self.night = self.night_av[constants.NIGHT_NIGHT]
            self._mtf_lut = None  # recalculate LUT


        if isinstance(self._mtf_lut, type(None)):
            if self.config.get('USE_NIGHT_COLOR', True):
                midtones = self.config.get('SCNR_MTF_MIDTONES', 0.55)
            else:
                if self.night_av[constants.NIGHT_NIGHT]:
                    # night
                    midtones = self.config.get('SCNR_MTF_MIDTONES', 0.55)
                else:
                    # day
                    midtones = self.config.get('SCNR_MTF_MIDTONES_DAY', 0.65)


            shadows_val = 0  # no clipping
            highlights_val = numpy.iinfo(scidata.dtype).max

            data_max = highlights_val

            range_array = numpy.arange(0, data_max + 1, dtype=numpy.float32)

            # these will result in 1.0 normalized values
            lut = (range_array - shadows_val) / (highlights_val - shadows_val)
            lut = ((midtones - 1) * lut) / (((2 * midtones - 1) * lut) - midtones)

            # back to real values
            lut = lut * data_max


            lut[lut < 0] = 0  # clip low end
            lut[lut > data_max] = data_max  # clip high end

            if scidata.dtype == numpy.uint16:
                numpy.rint(lut, out=lut)
            lut = lut.astype(scidata.dtype)  # this must come after clipping

            #logger.info('Min: %d, Max: %d', numpy.min(lut), numpy.max(lut))

            self._mtf_lut = lut


        b, g, r = cv2.split(scidata)


        mtf_g = self._mtf_lut[g]


        #stretch_elapsed_s = time.time() - mtf_start
        #logger.info('Stretch in %0.4f s', stretch_elapsed_s)

        return cv2.merge((b, mtf_g, r))


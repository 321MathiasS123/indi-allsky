"""Keep capture-time periods through the actual FITS ingestion methods."""
import ast
from datetime import date, datetime, timezone
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy
import psutil
import pytest

from indi_allsky import constants


@pytest.mark.parametrize('entry,captured', [('add', False), ('add', True), ('_add', False), ('_add', True)])
def test_fits_ingestion_preserves_capture_period_and_legacy_date(tmp_path, entry, captured):
    from astropy.io import fits

    # Execute both production methods and the real ImageData constructor;
    # replacing _add would hide a missing argument or undefined local there.
    source = Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {'ImageProcessor', 'ImageData'}]
    namespace = dict(constants=constants, numpy=numpy, cv2=cv2, Path=Path, time=time, psutil=psutil,
                     timezone=timezone, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    processor_class = namespace['ImageProcessor']
    processor = processor_class.__new__(processor_class)
    processor.config = {'TARGET_ADU': 100, 'TARGET_ADU_DAY': 50}
    processor.night_av = [1, 0]
    processor._detection_mask_dict = {}
    processor._check_astro_darkness = lambda: None
    processor.stack_count = 1
    processor.image_list = []
    processor._keogram_store_p = tmp_path
    processor._max_bit_depth = 16
    fallback_date, capture_date = date(2026, 10, 8), date(2026, 10, 7)
    processor._dateCalcs = SimpleNamespace(calcDayDate=Mock(return_value=fallback_date))
    camera = SimpleNamespace(id=1, name='test', uuid='test', owner='', location='',
                             lensFocalLength=2.1, lensFocalRatio=2.0, data={})
    received = datetime(2026, 10, 9, 0, 30, tzinfo=timezone.utc)
    pixels = numpy.arange(64, dtype=numpy.uint16).reshape(8, 8) + 32768
    filename = tmp_path / 'capture.fit'
    fits.PrimaryHDU(pixels).writeto(filename)
    kwargs = {'capture_day_date': capture_date} if captured else {}

    frame = getattr(processor, entry)(filename, 20, 50, 1, received, 25, camera, **kwargs)
    try:
        assert frame.day_date == (capture_date if captured else fallback_date)
        assert frame.exp_date == received
        numpy.testing.assert_array_equal(frame.hdulist[0].data, pixels)
        assert processor.image_list == ([frame] if entry == 'add' else [])
        if captured:
            processor._dateCalcs.calcDayDate.assert_not_called()
        else:
            processor._dateCalcs.calcDayDate.assert_called_once_with(received)
    finally:
        frame.hdulist.close()

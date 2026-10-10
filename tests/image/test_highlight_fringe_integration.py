"""Optional fringe correction uses capture-time daylight in both render paths."""
import ast
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def processor():
    source = ast.parse((ROOT / 'indi_allsky/processing.py').read_text(encoding='utf-8'))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ImageProcessor')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'reduce_highlight_fringes')
    helper = Mock(side_effect=lambda image: image + 1)
    namespace = dict(math=math, reduce_highlight_fringes=helper)
    cls.body = [method]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'fringe-integration', 'exec'), namespace)
    obj = SimpleNamespace(config={}, focus_mode=False, astrometric_data={'sun_alt': 20},
                          image=np.zeros((2, 3, 3), dtype=np.uint16), night_av=[True, False])
    return obj, namespace['ImageProcessor'].reduce_highlight_fringes, helper


@pytest.mark.parametrize('highlight,transition,denoise,fringe', list(itertools.product((False, True), repeat=4)))
def test_switch_is_independent_of_other_feature_activation(processor, highlight, transition, denoise, fringe):
    obj, apply, helper = processor
    obj.config = {'HIGHLIGHT_PROTECTION': {'ENABLE': highlight, 'FRINGE_REDUCTION': fringe},
                  'TWILIGHT_TRANSITION': {'ENABLE': transition},
                  'IMAGE_DENOISE_DAY': 'star_aware' if denoise else ''}
    before = obj.image
    apply(obj)
    if highlight and fringe:
        helper.assert_called_once_with(before)
        np.testing.assert_array_equal(obj.image, before + 1)
    else:
        helper.assert_not_called()
        assert obj.image is before


@pytest.mark.parametrize('config', [{}, {'HIGHLIGHT_PROTECTION': {}},
                                    {'HIGHLIGHT_PROTECTION': {'ENABLE': True}}])
def test_older_configs_do_not_touch_image_or_require_astrometry(processor, config):
    obj, apply, helper = processor
    obj.config = config
    del obj.astrometric_data
    before = obj.image
    apply(obj)
    helper.assert_not_called()
    assert obj.image is before


@pytest.mark.parametrize('altitude,enabled', [(None, False), ('unknown', False), (float('nan'), False),
                                           (float('inf'), False), (-40, False), (-.01, False),
                                           (0, True), (20, True)])
@pytest.mark.parametrize('night_profile', [False, True])
def test_capture_sun_elevation_controls_daylight_not_processing_profile(processor, altitude, enabled, night_profile):
    obj, apply, helper = processor
    obj.config = {'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'FRINGE_REDUCTION': True}}
    obj.astrometric_data['sun_alt'] = altitude
    obj.night_av[0] = night_profile
    before = obj.image
    apply(obj)
    assert helper.call_count == int(enabled)
    if not enabled:
        assert obj.image is before


def test_focus_mode_skips_correction(processor):
    obj, apply, helper = processor
    obj.config = {'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'FRINGE_REDUCTION': True}}
    obj.focus_mode = True
    apply(obj)
    helper.assert_not_called()


@pytest.mark.parametrize('path,prefix', [('image.py', 'self.image_processor'),
                                      ('flask/views.py', 'image_processor')])
def test_worker_and_fits_preview_apply_correction_before_sharpening(path, prefix):
    source = (ROOT / 'indi_allsky' / path).read_text(encoding='utf-8')
    start = source.rfind('\n', 0, source.index(prefix + '.apply_gamma_correction()')) + 1
    end = source.index(prefix + '.finish_colour_precision()', start) + len(prefix + '.finish_colour_precision()')
    calls = []
    methods = ('apply_gamma_correction', 'reduce_highlight_fringes', 'sharpen', 'finish_colour_precision')
    obj = SimpleNamespace(**{name: (lambda name=name: calls.append(name)) for name in methods})
    exec(textwrap.dedent(source[start:end]), {'self': SimpleNamespace(image_processor=obj), 'image_processor': obj})
    assert calls == list(methods)

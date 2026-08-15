import pytest

from indi_allsky.panorama import buildPanoramaCropFilter
from indi_allsky.panorama import validatePanoramaAspectRatio


def test_full_panorama_needs_no_filter():
    assert buildPanoramaCropFilter(4096, 1024, 0, 0, 4096, 1024) == ''


def test_crop_inside_saved_panorama_edges():
    assert buildPanoramaCropFilter(4096, 1024, 200, 100, 1200, 600) == (
        'crop=w=1200:h=600:x=200:y=100'
    )


def test_crop_wraps_saved_panorama_edge():
    assert buildPanoramaCropFilter(4096, 1024, 3600, 100, 1000, 600) == (
        'split=2[pano_right_src][pano_left_src];'
        '[pano_right_src]crop=w=496:h=600:x=3600:y=100[pano_right];'
        '[pano_left_src]crop=w=504:h=600:x=0:y=100[pano_left];'
        '[pano_right][pano_left]hstack=inputs=2'
    )


def test_full_width_crop_can_move_the_output_seam():
    assert buildPanoramaCropFilter(4096, 1024, 200, 0, 4096, 1024) == (
        'split=2[pano_right_src][pano_left_src];'
        '[pano_right_src]crop=w=3896:h=1024:x=200:y=0[pano_right];'
        '[pano_left_src]crop=w=200:h=1024:x=0:y=0[pano_left];'
        '[pano_right][pano_left]hstack=inputs=2'
    )


@pytest.mark.parametrize(
    'crop,error_text',
    (
        ((-2, 0, 100, 100), 'X coordinate'),
        ((4096, 0, 100, 100), 'X coordinate'),
        ((0, -2, 100, 100), 'Y coordinate'),
        ((0, 0, 4098, 100), 'width'),
        ((0, 900, 100, 200), 'height'),
        ((1, 0, 100, 100), 'must be even'),
        ((0, 0, 101, 100), 'must be even'),
    ),
)
def test_invalid_crop_is_rejected(crop, error_text):
    with pytest.raises(ValueError, match=error_text):
        buildPanoramaCropFilter(4096, 1024, *crop)


def test_odd_source_dimensions_are_rejected_for_yuv420_video():
    with pytest.raises(ValueError, match='must be even'):
        buildPanoramaCropFilter(4095, 1024, 0, 0, 1000, 600)


@pytest.mark.parametrize(
    'aspect_ratio,width,height',
    (
        ('free', 1000, 600),
        ('16:9', 1920, 1080),
        ('9:16', 1080, 1920),
        ('1:1', 1080, 1080),
        ('4:5', 1080, 1350),
        ('4:3', 1600, 1200),
        ('3:4', 1200, 1600),
        ('18:9', 2000, 1000),
        ('9:18', 1000, 2000),
        ('19.5:9', 1170, 540),
        ('9:19.5', 540, 1170),
        ('20:9', 1200, 540),
        ('9:20', 540, 1200),
        ('21:9', 1260, 540),
        ('9:21', 540, 1260),
    ),
)
def test_supported_aspect_ratios(aspect_ratio, width, height):
    assert validatePanoramaAspectRatio(aspect_ratio, width, height) == aspect_ratio


def test_fixed_aspect_ratio_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match='do not match'):
        validatePanoramaAspectRatio('16:9', 1920, 1082)


def test_unknown_aspect_ratio_is_rejected():
    with pytest.raises(ValueError, match='Unsupported'):
        validatePanoramaAspectRatio('2.39:1', 1920, 804)

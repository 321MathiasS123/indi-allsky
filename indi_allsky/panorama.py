PANORAMA_ASPECT_RATIOS = {
    'free' : None,
    '16:9' : (16, 9),
    '9:16' : (9, 16),
    '1:1'  : (1, 1),
    '4:5'  : (4, 5),
    '4:3'  : (4, 3),
    '3:4'  : (3, 4),
    '18:9' : (2, 1),
    '9:18' : (1, 2),
    '19.5:9' : (13, 6),
    '9:19.5' : (6, 13),
    '20:9' : (20, 9),
    '9:20' : (9, 20),
    '21:9' : (7, 3),
    '9:21' : (3, 7),
}


def validatePanoramaAspectRatio(aspect_ratio, crop_width, crop_height):
    aspect_ratio = str(aspect_ratio)

    try:
        ratio = PANORAMA_ASPECT_RATIOS[aspect_ratio]
    except KeyError:
        raise ValueError('Unsupported panorama aspect ratio')

    if ratio:
        ratio_width, ratio_height = ratio
        if int(crop_width) * ratio_height != int(crop_height) * ratio_width:
            raise ValueError('Panorama crop dimensions do not match the selected aspect ratio')

    return aspect_ratio


def buildPanoramaCropFilter(source_width, source_height, crop_x, crop_y, crop_width, crop_height):
    source_width = int(source_width)
    source_height = int(source_height)
    crop_x = int(crop_x)
    crop_y = int(crop_y)
    crop_width = int(crop_width)
    crop_height = int(crop_height)

    if source_width < 2 or source_height < 2:
        raise ValueError('Panorama source dimensions must be at least 2 x 2 pixels')

    if any(v % 2 for v in (source_width, source_height, crop_x, crop_y, crop_width, crop_height)):
        raise ValueError('Panorama source and crop coordinates and dimensions must be even')

    if crop_x < 0 or crop_x >= source_width:
        raise ValueError('Panorama crop X coordinate is outside the source image')

    if crop_y < 0 or crop_y >= source_height:
        raise ValueError('Panorama crop Y coordinate is outside the source image')

    if crop_width < 2 or crop_width > source_width:
        raise ValueError('Panorama crop width is outside the source image')

    if crop_height < 2 or crop_y + crop_height > source_height:
        raise ValueError('Panorama crop height is outside the source image')

    if (
        crop_x == 0
        and crop_y == 0
        and crop_width == source_width
        and crop_height == source_height
    ):
        return ''

    crop_end_x = crop_x + crop_width
    if crop_end_x <= source_width:
        return 'crop=w={0:d}:h={1:d}:x={2:d}:y={3:d}'.format(
            crop_width,
            crop_height,
            crop_x,
            crop_y,
        )

    right_width = source_width - crop_x
    left_width = crop_width - right_width

    return (
        'split=2[pano_right_src][pano_left_src];'
        '[pano_right_src]crop=w={right_width:d}:h={height:d}:x={x:d}:y={y:d}[pano_right];'
        '[pano_left_src]crop=w={left_width:d}:h={height:d}:x=0:y={y:d}[pano_left];'
        '[pano_right][pano_left]hstack=inputs=2'
    ).format(
        right_width=right_width,
        left_width=left_width,
        height=crop_height,
        x=crop_x,
        y=crop_y,
    )

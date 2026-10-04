"""On-demand keograms from a camera's retained image time range."""

from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image
import numpy

from .keogram import KeogramGenerator


# Bound the width of the in-memory keogram, including during direct worker use.
MAX_FRAMES = 20000


def parse_range(start, end):
    """Keep datetime-local input in the same wall-clock time as capture dates."""
    try:
        start_date = datetime.strptime(start, '%Y-%m-%dT%H:%M')
        end_date = datetime.strptime(end, '%Y-%m-%dT%H:%M')
    except (TypeError, ValueError):
        raise ValueError('Choose a valid start and end date and time.') from None
    if end_date <= start_date:
        raise ValueError('The end must be later than the start.')
    return start_date, end_date


def image_query(model, camera_id, start, end):
    # Inclusive endpoints deliberately cross the usual dayDate/night boundary.
    # The ID breaks ties when multiple frames have the same capture timestamp.
    return model.query.filter(
        model.camera_id == camera_id,
        model.createDate >= start,
        model.createDate <= end,
        model.exclude.is_(False),
    ).order_by(model.createDate.asc(), model.id.asc())


def output_path(image_dir, task_id):
    # Task IDs isolate exports; existing scratch cleanup expires their previews.
    return Path(image_dir).joinpath('scratch', 'custom_keogram_{0:d}.jpg'.format(task_id))


def validate_frame_count(count):
    if count > MAX_FRAMES:
        raise ValueError('This range has too many images. Choose a shorter range (up to {0:,} images).'.format(MAX_FRAMES))


def generate(config, camera, entries, outfile, progress):
    # Export separately from the scheduled keograms, using the same processor.
    export_config = dict(config, IMAGE_FILE_TYPE='jpg')
    generator = KeogramGenerator(export_config)
    generator.h_scale_factor = config.get('KEOGRAM_H_SCALE', 100)
    generator.v_scale_factor = config.get('KEOGRAM_V_SCALE', 33)
    generator.crop_top = config.get('KEOGRAM_CROP_TOP', 0)
    generator.crop_bottom = config.get('KEOGRAM_CROP_BOTTOM', 0)
    generator.label = config.get('KEOGRAM_LABEL', True)

    result = {'frames': 0, 'skipped': 0, 'resized': 0, 'first': None, 'last': None}
    image_shape = None
    for index, entry in enumerate(entries, start=1):
        validate_frame_count(index)
        try:
            with Image.open(entry.getFilesystemPath()) as source:
                data = cv2.cvtColor(numpy.array(source.convert('RGB')), cv2.COLOR_RGB2BGR)
        except OSError:
            # Retention may remove originals after the request has been queued.
            result['skipped'] += 1
        else:
            if image_shape is None:
                image_shape = data.shape[:2]
            elif data.shape[:2] != image_shape:
                height, width = image_shape
                source_height, source_width = data.shape[:2]
                if abs((source_width / source_height) / (width / height) - 1) > 0.01:
                    raise ValueError('The image shape changed during this range. Choose images with the same field of view.')
                # Day/night binning can change resolution without changing the sky view.
                data = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA)
                result['resized'] += 1

            generator.processImage(data, entry.createDate.timestamp())
            if result['frames'] == 0:
                # The shared generator starts with an unused column.
                generator.keogram_data = generator.keogram_data[:, 1:]
                result['first'] = entry.createDate.isoformat(sep=' ', timespec='seconds')
            result['frames'] += 1
            result['last'] = entry.createDate.isoformat(sep=' ', timespec='seconds')

        # Throttle database writes while counting unreadable frames as progress.
        if index % 25 == 0:
            progress(dict(result))

    if not result['frames']:
        raise ValueError('No readable local images remain in this range. Choose a range whose original images are still saved on this server.')
    # Very short ranges must still produce at least one output column.
    if int(result['frames'] * generator.h_scale_factor / 100) < 1:
        generator.h_scale_factor = 100
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    generator.finalize(outfile, camera)
    progress(dict(result))
    return result

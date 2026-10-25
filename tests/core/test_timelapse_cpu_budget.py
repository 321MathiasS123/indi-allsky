"""Exercise the real command builder without a Flask app or encoder process."""
import ast
import logging
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'indi_allsky' / 'timelapse.py'


@pytest.mark.parametrize('setting,expected', [(None, 1), (0, 0), (2, 2), ('invalid', 1)])
def test_timelapse_bounds_input_filter_and_output_threads(tmp_path, setting, expected):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'TimelapseGenerator')
    config = {'IMAGE_FILE_TYPE': 'jpg'}
    if setting is not None:
        config['VIDEO_CPU_LIMIT'] = setting
    source = tmp_path / 'source.jpg'
    source.write_bytes(b'image')
    video = tmp_path / 'video.mp4'
    preprocessor = SimpleNamespace(main=Mock(), seqfolder=tmp_path)
    def encode(*args, **kwargs):
        video.write_bytes(b'video')
        return SimpleNamespace(stdout=b'')
    run = Mock(side_effect=encode)
    namespace = {'os': os, 'time': time, 'Path': Path,
                 'subprocess': SimpleNamespace(run=run, PIPE=subprocess.PIPE,
                                               STDOUT=subprocess.STDOUT,
                                               CalledProcessError=subprocess.CalledProcessError),
                 'logger': logging.getLogger(__name__),
                 'timelapse_preprocessor': SimpleNamespace(standard=lambda config: preprocessor)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    generator = namespace['TimelapseGenerator'](config)
    generator.vf_scale = '-2:2160'
    generator.ffmpeg_extra_options = '-metadata title=sunset'
    generator.generate(video, [source])
    command = run.call_args.args[0]
    input_index = command.index('-i')
    thread_indices = [index for index, value in enumerate(command) if value == '-threads']
    if expected:
        assert len(thread_indices) == 2
        assert thread_indices[0] < input_index < thread_indices[1]
        assert [command[index + 1] for index in thread_indices] == [str(expected)] * 2
        assert command[command.index('-filter_threads') + 1] == str(expected)
        assert command[command.index('-filter_complex_threads') + 1] == str(expected)
    else:
        assert thread_indices == []
        assert '-filter_threads' not in command
        assert '-filter_complex_threads' not in command
    assert command[command.index('-c:v') + 1] == 'libx264'
    assert command[command.index('-b:v') + 1] == '5000k'
    assert command[command.index('-vf') + 1] == 'scale=-2:2160'
    assert command[-1] == str(video)
    preprocessor.main.assert_called_once_with([source])

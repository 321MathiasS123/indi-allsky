"""Check the discovery publisher and its compatibility with completion events."""

import ast
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import paho.mqtt.enums
import pytest

from indi_allsky import constants
from indi_allsky.timelapse_completion import completion_payload, record_media


@pytest.fixture
def discovery():
    # Avoid loading the Linux camera/DBus application. Execute the real discovery
    # class; replace only configuration loading, hardware probes and publication.
    source = Path(__file__).resolve().parents[2] / 'misc/home_assistant_auto_discovery.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'HADiscovery')
    publish = Mock()
    namespace = dict(constants=constants, json=json, sys=sys, paho=paho,
                     time=SimpleNamespace(sleep=Mock()), publish=publish,
                     psutil=SimpleNamespace(disk_partitions=lambda: [], sensors_temperatures=lambda: {}),
                     logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    instance = namespace['HADiscovery'].__new__(namespace['HADiscovery'])
    instance.config = {'FISH2PANO': {'ENABLE': False}, 'MQTTPUBLISH': {
        'ENABLE': True, 'TRANSPORT': 'tcp', 'HOST': 'localhost', 'PORT': 1883,
        'USERNAME': '', 'PASSWORD': '', 'TLS': False, 'BASE_TOPIC': 'indi-allsky',
    }}
    instance.device_name = 'indi-allsky'
    instance._port = 1883
    instance.update_sensor_slot_labels = Mock()
    return instance, publish


def published_event(discovery, retain=True):
    instance, publish = discovery
    instance.main(retain=retain)
    messages = publish.multiple.call_args.args[0]
    events = [message for message in messages if '/event/' in message['topic']]
    assert len(events) == 1
    return events[0], json.loads(events[0]['payload']), messages


@pytest.mark.parametrize('base,device,suffix', [
    ('indi-allsky', 'indi-allsky', '001'),
    ('garden-allsky', 'Garden camera', 'garden'),
])
@pytest.mark.parametrize('retain', [False, True])
@pytest.mark.parametrize('panorama', [False, True])
def test_discovery_groups_event_with_existing_device(discovery, base, device, suffix, retain, panorama):
    instance, _ = discovery
    instance.config['MQTTPUBLISH']['BASE_TOPIC'] = base
    instance.config['FISH2PANO']['ENABLE'] = panorama
    instance.device_name = device
    instance.unique_id_base = suffix
    message, config, messages = published_event(discovery, retain)
    assert message['topic'] == f'homeassistant/event/{base}/indi_allsky_timelapse_complete/config'
    assert message['retain'] is retain
    assert config['name'] == 'Timelapse completed'
    assert config['state_topic'] == f'{base}/timelapse/complete'
    assert config['unique_id'] == f'indi_allsky_timelapse_complete_{suffix}'
    assert config['event_types'] == ['success', 'failed']
    existing_image = next(json.loads(item['payload']) for item in messages if '/image/' in item['topic'])
    assert config['device'] == existing_image['device']
    assert config['device']['identifiers'] == [device]
    assert len({json.loads(item['payload'])['unique_id'] for item in messages}) == len(messages)
    assert published_event(discovery, retain)[0] == message


@pytest.mark.parametrize('success', [False, True])
def test_real_completion_payload_matches_declared_event_types(discovery, tmp_path, success):
    _, config, _ = published_event(discovery)
    task = SimpleNamespace(id=42, state=SimpleNamespace(value='Success'), data={
        'kwargs': {'camera_id': 7, 'timespec': '20261008', 'night': True},
    })
    media = tmp_path / 'timelapse.mp4'
    media.write_bytes(b'local video')
    record_media(task, {'timelapse': (SimpleNamespace(success=success), media)})
    payload = json.loads(json.dumps(completion_payload(task, {task.id: task})))
    # MQTT event consumers use event_type and retain the remaining JSON fields
    # as event attributes. Existing raw-topic automations still use status.
    assert payload['event_type'] in config['event_types']
    assert payload['event_type'] == payload['status'] == ('success' if success else 'failed')
    assert payload['event'] == 'timelapse_complete'
    assert payload['camera_id'] == 7
    assert payload['date'] == '2026-10-08'
    assert payload['period'] == 'night'
    assert payload['outputs']['timelapse'] == payload['status']
    assert payload['completed_at']


def test_disabled_mqtt_does_not_publish_discovery(discovery):
    instance, publish = discovery
    instance.config['MQTTPUBLISH']['ENABLE'] = False
    with pytest.raises(SystemExit):
        instance.main()
    publish.multiple.assert_not_called()

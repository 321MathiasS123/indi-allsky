# General
indi-allsky has the built-in capability to publish data from your all sky camera to an MQTT broker service.

You can find the settings related to the MQTT service in the configuration form.

## MQTT Broker Setup
If you need a local MQTT broker installed on your machine, you can configure Mosquitto automatically using [`indi-allsky-ctl`](indi-allsky-ctl):
```bash
indi-allsky-ctl setup-mqtt
```

## Home Assistant Auto-Discovery
Included is a utility to publish auto-discovery topics for Home Assistant. The discovery process only requires a few seconds to run and only needs to be executed once. Images and other sensors related to your allsky camera will automatically populate on your Home Assistant dashboard.

First, setup the MQTT integration in Home Assistant. After the integration is enabled, run the discovery command:

```bash
indi-allsky-ctl ha-discovery
```

*(Or from a git checkout: `source virtualenv/indi-allsky/bin/activate && ./misc/home_assistant_auto_discovery.py`).*

**If you run the script before the Home Assistant MQTT integration is setup, no entities will be displayed.** Simply run `indi-allsky-ctl ha-discovery` again once the integration is active.

## Topics
The default base topic is `indi-allsky/`. This can be overridden in the config section.
| Topic                  | Type      | Data |
| ---------------------- | --------- | ---- |
| indi-allsky/latest     | bytearray | Latest binary image |
| indi-allsky/sqm        | float     | SQM value |
| indi-allsky/stars      | int       | Number of stars detected |
| indi-allsky/exp_date   | str       | Exposure date and time |
| indi-allsky/exposure   | float     | Last camera exposure |
| indi-allsky/bin        | int       | Camera bin value |
| indi-allsky/temp       | float     | Last camera temperature reading |
| indi-allsky/sunalt     | float     | Sun altitude |
| indi-allsky/moonalt    | float     | Moon altiude |
| indi-allsky/moonphase  | float     | Moon phase % |
| indi-allsky/mooncycle  | float     | Moon cycle % |
| indi-allsky/moonmode   | bool      | True if indi-allsky has detected moon mode |
| indi-allsky/night      | bool      | True if indi-allsky is in night mode |
| indi-allsky/latitude   | float     | Latitude |
| indi-allsky/longitude  | float     | Longitude |
| indi-allsky/sidereal_time | str    | Sidereal time |
| indi-allsky/kpindex       | float  | Global Kp-index |
| indi-allsky/ovation_max   | int    | Max Aurora Ovation score for location |
| indi-allsky/smoke_rating  | str    | Smoke Rating |
| indi-allsky/aircraft      | int    | Count of visible Aircraft (ADS-B) |
| indi-allsky/cpu/user      | float  | User CPU Usage (percent) |
| indi-allsky/cpu/system    | float  | System CPU Usage |
| indi-allsky/cpu/nice      | float  | Nice CPU Usage |
| indi-allsky/cpu/iowait    | float  | IO Wait CPU Usage |
| indi-allsky/cpu/total     | float  | Total CPU Usage |
| indi-allsky/memory/user   | float  | Memory Usage (percent) |
| indi-allsky/memory/cached | float  | Cached Memory Usage |
| indi-allsky/memory/total  | float  | Total Memory Usage |
| indi-allsky/disk/root     | float  | Root Filesystem Usage (percent) |
| indi-allsky/disk/MOUNTPOINT | float  | Filesystem Usage (for each mountpoint) |
| indi-allsky/temp/SYS/LABEL  | float  | Temperature<br/>Example for RPi 3: `indi-allsky/temp/cpu_thermal/0`<br />The System page will show the same names used in MQTT |

| Topic                  | Type      | Data |
| ---------------------- | --------- | ---- |
| indi-allsky/sensor_temp_0 | float  | Camera temperature |
| indi-allsky/sensor_temp_1<br>indi-allsky/sensor_temp_2<br>...<br>indi-allsky/sensor_temp_9 | float  | Reserved |
| indi-allsky/sensor_temp_10<br>indi-allsky/sensor_temp_11<br>...<br>indi-allsky/sensor_temp_29 | float  | System Temperature Data |

| Topic                  | Type      | Data |
| ---------------------- | --------- | ---- |
| indi-allsky/sensor_user_0 | float  | Camera temperature |
| indi-allsky/sensor_user_1 | float  | Dew Heater Duty Cycle |
| indi-allsky/sensor_user_2 | float  | Dew Point |
| indi-allsky/sensor_user_3 | float  | Frost Point |
| indi-allsky/sensor_user_4 | float  | Fan Duty Cycle |
| indi-allsky/sensor_user_5 | float  | Heat Index |
| indi-allsky/sensor_user_6 | float  | Wind direction (degrees) |
| indi-allsky/sensor_user_7 | float  | SQM Magnitude |
| indi-allsky/sensor_user_8<br>indi-allsky/sensor_user_9<br> | float  | Reserved |
| indi-allsky/sensor_user_10<br>indi-allsky/sensor_user_11<br>...<br>indi-allsky/sensor_user_29 | float  | User sensor data |

## Local timelapse completion

When MQTT publishing is enabled, automatic day/night generation publishes a JSON
event to `<base topic>/timelapse/complete` (normally
`indi-allsky/timelapse/complete`). No additional indi-allsky setting is needed.
It works even when publishing image bytes is disabled.

The event is queued after the final local generation job: panorama when enabled,
otherwise the normal timelapse. It checks the exact keogram/startrail and video
tasks belonging to that batch, their generation results, and the local files.
It does not wait for remote uploads. Delivery uses the existing MQTT upload
workers, so a busy upload queue or broker outage can delay or prevent delivery.

Example successful night event:

```json
{
  "event": "timelapse_complete",
  "event_id": "timelapse-12345",
  "camera_id": 1,
  "date": "2026-10-08",
  "period": "night",
  "status": "success",
  "completed_at": "2026-10-09T06:42:15+00:00",
  "outputs": {
    "keogram": "success",
    "startrail": "success",
    "startrail_timelapse": "success",
    "timelapse": "success",
    "panorama": "success"
  },
  "failed_task_ids": []
}
```

`date` identifies the capture day/night, not the date the work finished.
`completed_at` is UTC. `status` is `success` only if all batch tasks succeeded
and every generated output exists locally and is nonempty; otherwise it is
`failed`. Optional startrail outputs skipped during daytime or because too few
frames qualified have output status `skipped`. A disabled panorama is omitted.
A failed task can have no output entries; its ID appears in `failed_task_ids`.
A startrail encoder failure is reported even when its combined keogram task
succeeded.

Events are not retained, so reconnecting does not replay an old completion.
They are not a durable notification history: Home Assistant must be connected
to receive them. MQTT QoS can redeliver a message; use `event_id` to deduplicate
if your action requires this. Manual regenerations and mini-timelapses do not
emit this automatic-batch event. A worker interrupted before reaching its final
job does not emit a successful completion.

In Home Assistant, use an MQTT trigger filtered to successful completion:

```yaml
alias: Allsky local timelapses ready
triggers:
  - trigger: mqtt
    topic: indi-allsky/timelapse/complete
    payload: success
    value_template: "{{ value_json.status }}"
actions:
  - action: persistent_notification.create
    data:
      title: Allsky timelapses ready
      message: >-
        Camera {{ trigger.payload_json.camera_id }}:
        {{ trigger.payload_json.date }} {{ trigger.payload_json.period }}
        finished locally at {{ trigger.payload_json.completed_at }}.
mode: queued
```

Adjust the topic if you use a different MQTT base topic. Filter on
`trigger.payload_json.camera_id` or `period` in a condition when desired.
Use `payload: failed` in a second automation to receive failure notifications.

## Local broker
Included in the repository is a script that an quickly deploy a Mosquitto MQTT broker to your system.

The script is at `./misc/setup_mosquitto.sh`.  The script will ask for a password for the `indi-allsky` mqtt user.

Mosquitto will be setup with the following ports:
| Port | Service |
| ---- | ------- |
| 1883 | mqtt no encryption (loopback only) |
| 8883 | mqtt with encryption |
| 8080 | websockets no encryption (loopback only) |
| 8081 | websockets with encryption |

## MQTT clients
You can use a mqtt client such as `MQTT Dash` on Android to subscribe to a local mosquitto server and view the data published in real-time.
https://play.google.com/store/apps/details?id=net.routix.mqttdash

[[https://github.com/aaronwmorris/indi-allsky/blob/master/content/screenshot_mqtt_dash.jpg | width=300px]]

### CLI
You may subscribe to topics from the CLI to debug the information being passed.

```
mosquitto_sub -d -V mqttv5 -h localhost -p 1883 -u mqtt_username -P password123 -t 'indi-allsky/exposure'
```

Certificate validation will likely fail for `mosquitto_sub`, therefore it is easier to subscribe to the non-TLS port.
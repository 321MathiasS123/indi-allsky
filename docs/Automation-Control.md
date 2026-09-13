# Home Assistant control

The automation API starts finite incremental sync runs and requests coordinated
capture recovery or reboot. It is disabled until a dedicated token and source
network are configured. It does not enable per-image uploads.

This branch starts from main. Production combines these features:

- `feature/home-assistant-control`: API, recovery helper, shutdown handling, HA package.
- `feature/syncapi-on-demand`: incremental sync, including machine-readable terminal reasons.
- `feature/latest-capture-health-warning`: the shared Latest-page capture-health calculation.

Without the optional sync or capture-health feature, status reports that capability
as unavailable and the corresponding operation cannot run. The API uses the same
successful-image timestamp, error information, and capture expectations as the
Latest page. Its response embeds that page's health result unchanged. HA recovery
waits at least eight minutes (longer for long exposures), and additionally avoids
recovery during boot, maintenance, and dark-library capture.

## Install on the Pi

1. Update production. First inspect the actual installed capture service and any
   overrides, and the forced command associated with the old HA SSH key. The
   repository alone cannot establish how that old reboot command behaved.
2. Run the installer as the normal indi-allsky user, using the installation's
   existing virtual environment. Supply the HA host's actual IP, not the NAS IP:

   ```sh
   virtualenv/indi-allsky/bin/python3 misc/install_automation.py \
     --ha-address HA_IP_ADDRESS --enable-recovery
   ```

   Add `--restart-indiserver` only if this Pi runs the local camera driver that
   should restart along with capture. Otherwise recovery restarts capture alone.
   `--config` supports a non-default Flask configuration path.

   The installer backs up Flask configuration, generates a separate random token,
   restricts it to the HA address, installs the helper, and adds a capture-service
   shutdown override. It reloads systemd definitions but does **not** restart
   capture, change the running sync, restart HA, or reboot anything.

3. Copy the generated private `homeassistant-secret.yaml` entry into HA's
   `secrets.yaml`. Do not commit this file, put the token in a URL, reuse the Flask
   session secret, or expose it in screenshots/logs.
4. Restart the configured Gunicorn **web** service to load the token. Capture
   continues separately. Verify the status endpoint before enabling any automation.
5. The new shutdown code takes effect after the capture service next starts.
   For its first restart after updating, an already-running old capture process
   still has its old Python code, although the new systemd grace period applies.

The shutdown default is 180 seconds for coordinated worker cleanup and 240 seconds
for systemd's final process-group deadline. Increase both together for unusually
long exposures; the helper service and command timeouts must also exceed them.
Normal reboots should finish cleanup. A stuck worker may still need termination;
that is reported as incomplete cleanup during requested recovery, never falsely
marked as a clean shutdown. Unexpected crashes retain their original warning.

The web server remains unprivileged. A fixed user service manages configured local
units and uses the existing login1/polkit reboot permission. Requests cannot name
arbitrary units or shell commands. Network checks use Flask's effective client IP;
the deployment's reverse proxy must overwrite forwarded client headers and the
Gunicorn socket must not be exposed directly to untrusted clients.

## Home Assistant package

Copy `examples/homeassistant/indi_allsky.yaml` to HA's packages directory. Enable
packages under the existing `homeassistant:` section if needed:

```yaml
homeassistant:
  packages: !include_dir_named packages
```

Preserve existing configuration; do not add a second `homeassistant:` key. Adjust
the Pi/NAS addresses and entity IDs in the package. This example uses the inspected
installation's `device_tracker.ds1821_1` and `sensor.indi_allsky_exposure_date`.
Both connections use the installation's self-signed certificates. Prefer trusted
certificates; if installed in HA's container, enable verification and replace the
NAS probe's unverified SSL context with `ssl.create_default_context()`.

Run HA's configuration check, reload/restart the required integrations, and verify
the created entity IDs. Enable the two switches independently:

- **Allsky automatic NAS sync**: checks every five minutes, requires three minutes
  of NAS service availability, starts a finite run, and repeats six hours after
  completion while the NAS stays on. Network failures wait 30 minutes. Three
  consecutive failed runs, or a non-network failure, disable the switch and notify.
- **Allsky automatic capture recovery**: uses stale MQTT as an alarm, verifies the
  shared local health result, requests a capture restart, and waits ten minutes for
  a newer image. If permitted by the current capture schedule, one coordinated
  reboot follows. Another ten minutes verifies a changed boot ID and a newer image.

The NAS probe returns OFF for expected connection failures, so powering off the
NAS does not generate failed-upload attempts or failed probe commands. A ready
web service is only a preliminary check: the Pi still verifies the real SyncAPI
connection and applies its existing bounded file retries.

The existing **Allsky - reboot stalled camera** SSH automation should remain
disabled when enabling the replacement. Do not enable both watchdogs. The example
switches start off on first installation and restore their previous state later.
The last recovery incident is also restored, preventing replay after an HA restart.
The API additionally enforces ten minutes between service recoveries and an hour
between reboot requests, including across Pi reboots.

Manual sync cancellation is respected: start **Sync now** in indi-allsky when ready
to resume. A service-maintenance interruption remains eligible for automatic
catch-up. Switching automatic sync off prevents future starts; it does not cancel
an already-running transfer. A running transfer can be cancelled in indi-allsky or
through the cancel endpoint.

If HTTP is unavailable, this package does not use the old SSH reboot command. The
API cannot recover an unreachable web server or a wedged OS. An independent
last-resort watchdog remains a separate deployment choice. HA restarts during an
incident do not automatically repeat that incident; inspect the Pi if it remains
stalled, and clear the last-recovery helper only when deliberately retrying.

## API contract

All paths start with `/indi-allsky/automation`. Every request, including reads,
requires `Authorization: Bearer <dedicated token>` from an allowed source address.
Browser sessions and `LOGIN_DISABLED` do not bypass this authentication.

| Method/path | Body | Result |
| --- | --- | --- |
| GET `/status` | None | Local capture health, sync progress, recovery operation and boot ID |
| POST `/sync/start` | `request_id`, optional `types` | 202 with task ID; an existing run is reused |
| POST `/sync/cancel` | `task_id` | Current sync status |
| POST `/system/recover` | `request_id`, optional `require_stale` | 202 with durable recovery operation |
| POST `/system/reboot` | Same | 202; changed boot ID confirms reboot |

Recovery requires a locally confirmed stall by default. `require_stale: false`
permits a deliberate manual operation, or HA's reboot escalation after a verified
stall and an unsuccessful restart. The HA escalation rechecks capture schedule,
pause/dark-capture status, maintenance, and its enable switch before rebooting.

Use a stable request ID when retrying a request whose response was lost. The most
recent request for each recovery action is persisted, as is the most recent sync
request. Completion of a service-control job only means systemd completed the job;
HA separately confirms that new images resumed. The helper is never enabled at
boot and never autonomously replays an old reboot request.

Sync terminal `reason` values include `connection`, `certificate`, `authentication`,
`receiver`, `configuration`, `unexpected`, `manual_cancel`, `configuration_changed`,
and `maintenance`. HTTP receiver rejection is deliberately not automatically
retried: it can represent full storage or a receiver-side error. `failure_streak`
counts consecutive failed runs and resets on a non-failed terminal run.

## Verification before enabling recovery

Tests cover token/network enforcement, duplicate and conflicting requests,
maintenance versus manual cancellation, shutdown deadlines, failed service jobs,
boot reconciliation, and shared capture-health behavior. Linux systemd behavior
still requires validation on the Pi: verify a normal service restart, then a
planned reboot, checking that capture resumes, clean shutdowns have no abnormal
shutdown warning, and interrupted sync resumes without copying acknowledged files.
Use HA's configuration validator before activating the package; ordinary YAML and
template tests cannot validate the installed HA integration runtime.

# On-demand synchronization

Use this mode when the receiving indi-allsky installation is only occasionally available, for example a NAS that is normally powered off.

Update **both the sending Pi and the receiver** to a version containing this feature. The receiver update adds an authenticated file lookup and fixes interrupted image transfers and multiple mini timelapses in the same day/night period. No database migration or reset of existing sync records is required. For a Docker receiver built from source, rebuild and recreate its application container from the updated source while preserving the database and media volumes.

In the Pi's configuration, open **SyncAPI**, enable Sync API, and choose **On demand**. Keep the existing working URL, account, API key, and certificate settings. Save the configuration and wait for the service reload; changing the synchronization mode requests a reload automatically.

When the NAS is available, select the desired media types in **On-demand synchronization** and press **Sync now**. The run happens in the background and continues after the page is closed. Images, panoramas, timelapses, mini timelapses, keograms, star trails and their videos are selected by default. RAW and FITS files are optional. Camera metadata, associated thumbnails, custom metadata and the long-term keogram samples attached to images are included.

The run covers retained local-camera data, including images older than thirty days. Completed files from the last ten minutes are deferred. A fixed upper bound prevents ongoing capture from extending the run indefinitely. Hidden cameras and unfinished generated media are excluded. Automatic image sampling, waiting for S3, and empty-file uploads do not apply to the manual archive operation.

Pending media are transferred oldest first by creation time across all selected media types. Each parent file is followed by its associated thumbnail before the next media item is processed.

**Upload speed limit**, beside the two minute fields, caps archive uploads so they leave bandwidth for browsing and capture. Choose Unlimited (the existing default), 128/256/512 KiB/s, or 1/2/5/10 MiB/s. Save Configuration stores the limit for manual and scheduled runs; Sync now uses the currently selected speed and content even before saving. Cancel an active run before changing its speed. Automatic live uploads are unaffected. The cap includes multipart request bytes; actual throughput can be lower when the network or receiver is slower.

Large files show their current upload progress approximately every five seconds. These bytes have been read into the outgoing request; the completed file/byte totals increase only after the receiver acknowledges the file. The current-file counter resets on retry. Cancel is checked during uploads as well as between files; a blocked network operation still has to finish or time out.

The existing receiver authentication window allows about 20 minutes for an upload. If the selected cap alone would take longer for a file, the run stops with an explicit instruction to increase the speed limit, before uploading it. This preserves authentication checks and existing transfer records. A slower-than-expected connection can still exceed that window.

Successfully acknowledged items are skipped. If a previous upload arrived but its acknowledgement was lost, the Pi checks its identity, thumbnail reference, size and SHA-256 digest on the receiver and records the result without resending the media. Image/thumbnail completion is recorded together. Missing local files are reported as skipped and are never marked synchronized.

During a run, connection failures and timeouts are retried twice, after 5 and 15 seconds. Each retry checks whether the receiver already saved the file before sending it again. The panel shows the retry status; exhausted retries produce one failure summary with the file or camera, lookup/upload stage, and underlying error. Authentication, certificate and receiver rejection errors stop immediately. With the availability schedule disabled, there are no probes or per-image SyncAPI attempts between manual runs. Press **Sync now** again to continue a stopped run. **Cancel** interrupts retry waits and outgoing uploads; a blocked network operation must finish or time out first.

## Optional availability schedule

In the same panel, select **Automatically sync when available**, set **Check every** (default 10 minutes) and **Wait after availability** (default 3 minutes), select content types, and press the page's **Save Configuration** button. The switch, timings and selected content are validated and saved together with the configuration. Changed scheduling settings request a service reload so the scheduler uses the newly saved configuration. The schedule is disabled by default. Check intervals accept 1–1440 minutes; startup delays accept 0–1440 minutes. Existing schedules are preserved when upgrading. Cancel an active run before changing its schedule; saving unchanged settings does not interrupt that run.

After each check interval, the Pi uses the existing SyncAPI URL, username, API key and certificate setting for an authenticated, read-only camera lookup. This verifies that the receiver's application and database respond, rather than just its network interface. A receiver already supporting on-demand synchronization needs no further update. A camera not yet registered is also accepted as a readiness response; the first actual run registers it. Older receivers may log that initial missing-camera lookup.

If the receiver is unavailable, the Pi quietly waits another check interval. If available, it waits the startup delay and checks again before queuing one archive run. A zero delay starts the run after the first successful check. Completion or an exhausted connection retry starts a fresh check interval and startup-delay cycle. When no eligible files are pending, the cycle creates no probe or empty run. The service checks timers about every 13 seconds, so times are approximate; probes run in a separate thread and never make capture supervision wait for the network.

The schedule continues with the browser closed and after a Pi/service restart. A restart begins a fresh check interval, using existing transfer checkpoints when a new run starts. Only one run is admitted at a time; a manual run suspends availability checks. The most recently saved content selection is used for scheduled runs. Unsaved checkbox edits only affect **Sync now**.

**Cancel also pauses the schedule.** To resume, enable it and use **Save Configuration**. Authentication, certificate, unexpected response and non-connection transfer errors also pause it, with one warning instead of repeatedly trying the same broken configuration. Correct the problem, then enable and save again. A single status area displays transfer progress, the schedule's current phase, next action time and pause reason. Switching away from On demand mode or disabling Sync API prevents scheduled checks and uploads.

This scheduler is part of the existing on-demand worker; it adds no Home Assistant control API or reboot/recovery service.

The panel reports progress and the last outcome using the Pi's database. It never probes the NAS merely because the page is open. Existing successful-upload records are preserved, including uploads made in Automatic mode.

Those records belong to one destination. On demand mode records the configured URL/account and rejects silent reuse after a destination change; restore the original URL/account to continue. API-key rotation is allowed. Moving to a different server requires a separate, deliberate migration of the transfer records.

This is an incremental media archive, not a filesystem mirror or a backup of the Pi's configuration. It does not recheck previously acknowledged files for manual deletion/corruption on the NAS, detect a replacement receiver database at the same URL, or track later edits to acknowledged media. Removing expired files on the Pi does not delete their NAS copies.

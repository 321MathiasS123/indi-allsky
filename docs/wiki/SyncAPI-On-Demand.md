# On-demand synchronization

Use this mode when the receiving indi-allsky installation is only occasionally available, for example a NAS that is normally powered off.

Update **both the sending Pi and the receiver** to a version containing this feature. The receiver update adds an authenticated file lookup and fixes interrupted image transfers and multiple mini timelapses in the same day/night period. No database migration or reset of existing sync records is required. For a Docker receiver built from source, rebuild and recreate its application container from the updated source while preserving the database and media volumes.

In the Pi's configuration, open **SyncAPI**, enable Sync API, and choose **On demand**. Keep the existing working URL, account, API key, and certificate settings. Save the configuration and wait for the service reload; changing the synchronization mode requests a reload automatically.

When the NAS is available, select the desired media types in **On-demand synchronization** and press **Sync now**. The run happens in the background and continues after the page is closed. Images, panoramas, timelapses, mini timelapses, keograms, star trails and their videos are selected by default. RAW and FITS files are optional. Camera metadata, associated thumbnails, custom metadata and the long-term keogram samples attached to images are included.

The run covers retained local-camera data, including images older than thirty days. Completed files from the last ten minutes are deferred. A fixed upper bound prevents ongoing capture from extending the run indefinitely. Hidden cameras and unfinished generated media are excluded. Automatic image sampling, waiting for S3, and empty-file uploads do not apply to the manual archive operation.

Successfully acknowledged items are skipped. If a previous upload arrived but its acknowledgement was lost, the Pi checks its identity, thumbnail reference, size and SHA-256 digest on the receiver and records the result without resending the media. Image/thumbnail completion is recorded together. Missing local files are reported as skipped and are never marked synchronized.

An unavailable receiver or a failed transfer stops the run and produces one failure summary. There are no automatic retries, availability probes, or per-image SyncAPI attempts between manual runs. Press **Sync now** again to continue. **Cancel** stops further requests after the current request finishes or times out. Service restarts do not automatically resume a run.

The panel reports progress and the last outcome using the Pi's database. It never probes the NAS merely because the page is open. Existing successful-upload records are preserved, including uploads made in Automatic mode.

Those records belong to one destination. On demand mode records the configured URL/account and rejects silent reuse after a destination change; restore the original URL/account to continue. API-key rotation is allowed. Moving to a different server requires a separate, deliberate migration of the transfer records.

This is an incremental media archive, not a filesystem mirror or a backup of the Pi's configuration. It does not recheck previously acknowledged files for manual deletion/corruption on the NAS, detect a replacement receiver database at the same URL, or track later edits to acknowledged media. Removing expired files on the Pi does not delete their NAS copies.

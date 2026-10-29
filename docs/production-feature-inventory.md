# Production feature inventory and upstream status

Last functional review: **29 October 2026, Europe/Berlin**.

The revisions and counts below describe the last reviewed functional snapshot. Later documentation-only commits do not change that baseline. Update this inventory when production functionality or upstream status changes, following the maintenance instructions at the end.

Production contains substantial changes to calibration, exposure control, image rendering, capture lifecycle and archive synchronization. Many earlier contributions have reached upstream, including ASI676MC purple-frame handling, mini-timelapse improvements, VirtualSky tilt and calibration, and several image-processing fixes.

**For issue reproduction, turning off the optional features does not make production equivalent to upstream.** Runtime dark selection and high-precision colour rendering still differ, and capture ordering, recovery and resource scheduling also change how the installation behaves.

## Revisions and scope

| Reference | Revision | Meaning |
| --- | --- | --- |
| Audited production code | `29ff82407661493fb0ffd467d8d178565cb13aed` | Functional snapshot, last code commit 28 October |
| Audited upstream main | `938bf860b28a472ed71e801d75b7ec9a230b6fef` | Upstream revision checked at the last review |
| Last upstream main included in audited production | `18b8407d31843f3df9c69f8b5b16443f7df65c82` | Also the local main branch at audit time |
| Starting parent of first production-specific integration | `254379ebd3c34c6f4f6b272ffbd4afa91aa0f05a` | Upstream revision from 19 August |

The earliest surviving production-specific first-parent commit is [`1a90697c`](https://github.com/321MathiasS123/indi-allsky/commit/1a90697cbf8206c2e3e931f0d39ba479abd4fc65), dated **22 August 2026**, integrating the mini-timelapse overhaul. Dark automation followed the same day. Git does not permanently record a branch creation event, so this is the practical beginning of the production history. Some integrated development began earlier.

At the last functional review, production had **507 commits not in audited upstream main**, including 272 non-merge commits. These are not 507 separate features: the history includes merges, squashes, duplicate cherry-picks, reversions and later upstream equivalents. Relative to the last incorporated upstream revision, the remaining delta spanned 221 files, including extensive tests and generated assets.

At that review, upstream was two commits ahead of the shared revision: the sensor display-units change and its merge, [PR #3304](https://github.com/aaronwmorris/indi-allsky/pull/3304). That was **upstream work not yet included in production**, rather than a custom feature.

“Still custom” below means a functional difference remains in the checked production tree. “Upstream” means the functionality is in the checked upstream main; it does not imply availability in older releases. Defaults refer to source defaults. This inventory does not establish the revision or enabled settings currently running on the camera.

## Feature status at a glance

| Functionality | Status on 29 October | Main consequence |
| --- | --- | --- |
| Guided dark-library builder and management | Still custom; [#3112 open](https://github.com/aaronwmorris/indi-allsky/pull/3112) | Different calibration creation, coverage, maintenance and capture workflow |
| Closest-temperature dark and BPM selection | Still custom | Different calibration masters can be chosen for the same light frame |
| Dark-master integrity and temperature metadata | Still custom | Newly generated masters and their eligibility can differ |
| Fixed or exposure-matched dark cooldown | Still custom | Different dark capture timing and sensor temperature history |
| Highlight protection and compensated rendering | Still custom | Exposure, gain, clipping, shadows and gamma |
| Twilight prediction and output feedback | Still custom | Different dawn and dusk exposure response |
| Bounded Moon exception and Moon-night control | Still custom | Different Moon-night clipping, exposure and gain |
| Smooth day/night transition | Still custom | Capture settings and rendering vary continuously through twilight |
| Star-aware denoise and sensor-defect repair | Still custom | Different noise, detail, star and hot/cold-pixel appearance |
| Optional VirtualSky and BPM denoise assistance | Still custom | Additional source/defect confirmation |
| Higher colour precision and saturation shoulder | Still custom; no master opt-in | Ordinary rendered pixels can differ with optional features off |
| Fresh exposure feedback ahead of rendering | Still custom | Different exposure response during processing backlog |
| Capture-period ordering and orderly drain | Still custom | Final frames, date grouping, reload and shutdown behavior |
| Actual-frame watchdog with adaptive option | Still custom | Different stall detection and automatic recovery |
| Batch priority and CPU reservation | Still custom | Different contention and timelapse generation time |
| Background compressed FITS saving | Still custom | Different capture/render latency and FITS availability time |
| Archive SyncAPI and availability scheduling | Still custom; [#3267 open](https://github.com/aaronwmorris/indi-allsky/pull/3267) | Different synchronization, retry and receiver behavior |
| Custom time-range keograms | Still custom | Additional export workflow |
| Gallery panorama switch and toolbar | Still custom; [#3225 open](https://github.com/aaronwmorris/indi-allsky/pull/3225) | Different gallery navigation and actions |
| VirtualSky mirroring and related planet fixes | Still custom; [#3201 open](https://github.com/aaronwmorris/indi-allsky/pull/3201) | Different overlay geometry and mirror-aware solving |
| Latest capture-health warning | Still custom | Additional warning states and polling |
| Local-camera DST freshness and VirtualSky time correction | Still custom; [#3309 open](https://github.com/aaronwmorris/indi-allsky/pull/3309) | Different behavior across clock changes |
| Automatic timelapse MQTT completion and HA discovery | Still custom | Additional completion events for automations |
| Chart sizing at fractional browser zoom | Still custom; [#3298 open](https://github.com/aaronwmorris/indi-allsky/pull/3298) | Different chart layout on affected browsers |
| Mini-timelapse overhaul, bitrate and preview improvements | Upstream | Shared functionality on sufficiently recent versions |
| Loop playback improvements | Upstream | Shared buffering, cancellation and navigation |
| VirtualSky tilt, pointing and optional lens calibration | Upstream, with custom mirroring extensions remaining | Base solver is shared; full local behavior is not identical |
| ASI676MC purple handling and later calibration recovery | Upstream | Implementation and ASI documentation have no current residual |
| Gamma, MTF cache and daytime white-balance fixes | Upstream, with other rendering changes remaining | Individual fixes are shared; the whole pipeline still differs |
| Cardinal directions, map coordinates and notification fixes | Upstream | Shared application behavior, apart from remaining custom features |

## Calibration and dark-library changes

### Guided dark-library builder

**Production since 22 August; consolidated in late August and on 19 September. Still custom.**

The Dark Library page provides a guided alternative to the original dark command line. It derives capture plans from the configured day, night and Moon profiles and the camera’s actual capabilities, including gain range, discrete gain or ISO choices, exposure strategy, binning, image depth and observed frame dimensions.

The builder:

- Evaluates existing paired dark/BPM coverage and distinguishes usable, coarse, different-temperature, incompatible and missing coverage.
- Offers strategies to add missing sets, refresh recommended sets, rebuild profiles or use a custom plan. Existing usable masters can be retained.
- Supports custom gain and exposure lists, their combinations, longest-exposure limits, capture ordering, and estimates of time and storage.
- Provides Precise, Balanced and Fast spacing for continuous automatic-gain coverage. The source defaults use 10 source frames per master, with 3–50 permitted.
- Uses a per-camera temperature-matching preference for coverage recommendations, normally 5°C. It supports camera temperature, a configured CCD temperature script and sensorless operation.
- Can repeat night capture groups as temperature falls, optionally ending after a group at or below a target temperature.
- Validates the current camera, configuration, capabilities and temperature source before capture. Changed assumptions require reviewing the plan again.
- Learns actual received FITS dimensions for binning, preserving those learned dimensions across reconnects.
- Supports average stacking and sigma clipping for RAW input. RGB inputs use the supported average workflow, with validation of incompatible profile settings.

The camera must be covered. The managed run coordinates with normal capture, pauses it at the appropriate boundary, reports progress and cancellation, and restores the prior lifecycle state after completion or failure.

**Why this matters:** master creation, gain/exposure coverage and the sensor’s temperature during dark acquisition can differ from stock installations, even when the final light-frame settings look the same.

Sources: [Dark Calibration Frames guide](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/docs/wiki/Dark-Calibration-Frames.md), [planner](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/dark_library.py), [automation](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/dark_automation.py).

### Library maintenance and interrupted runs

The expanded page groups entries by camera, image profile, temperature and master set. It supports individual or grouped selection, reversible activation/deactivation, preview and deletion. Darks and their corresponding maps are managed together.

Completed dark/BPM pairs are checkpointed. Replacement masters are staged inactive before verified activation; interrupted runs preserve completed pairs and remove incomplete artifacts. The original CLI remains separate from the builder’s staging and rollback behavior.

Additional checks reject empty or all-zero masters, use the configured sensor depth for BPM thresholds, preserve genuine 0°C values and sensorless markers, and keep temperature metadata consistent across filenames, database records, manifests and FITS headers. Geometry restoration resets the unbinned frame before applying binning to avoid leftover driver ROI state.

**Why this matters:** these changes affect which masters exist, are active and are valid. Deactivating a set is also a local UI capability that may not exist on an issue reporter’s installation.

### Runtime dark and BPM temperature selection

**Developed 21 August and integrated with the original dark automation. Still custom. High relevance to image-processing reproduction.**

Production preserves the camera, active status, bit-depth and binning filters and the existing “gain at least capture gain” and “exposure at least capture exposure” requirements. It then prefers:

1. The lowest sufficient gain.
2. The lowest sufficient exposure.
3. The nearest known temperature in absolute distance, whether warmer or colder.
4. The newest matching master as a tie-breaker.

Unknown temperatures follow known temperatures. Upstream first searches for a same-temperature or warmer master within +5°C, then falls back to its hottest compatible master.

Thus **the same light frame and the same library can select different darks or BPMs in production and upstream**. This can affect calibrated signal, fixed-pattern residuals, noise, apparent colour and exposure feedback. It does not establish the cause of any specific past colour problem.

The builder’s configurable temperature tolerance is a coverage/recommendation preference; it is **not** a strict maximum-distance limit in the runtime calibration query.

Source: [processing.py calibration selection](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/processing.py).

### Cooldown between dark source frames

**Integrated 19 September. Still custom.**

The builder can wait a fixed duration or a duration equal to the exposure after each source frame. The source default is zero delay. The final source frame also receives its cooldown, overlapping that wait with master creation so the next set waits for both operations to finish. Time estimates include the cooldown. Original CLI runs do not inherit this builder-added delay.

This is the production implementation. The separate experiment with temperature-controlled cooldown and temperature-check exposures is not in production.

## Exposure and image-rendering changes

### Highlight protection

**First integrated 2 October, with substantial tuning through 11 October. Still custom. Source default off.**

Highlight protection supplements average-brightness exposure control with measurements of the largest connected near-clipped patches in the ADU mask. It tracks all-channel and any-channel clipping separately and meters before dark subtraction, so calibration cannot hide sensor clipping. Colour cameras retain colour metering even when the displayed output is grayscale.

It can reduce exposure or gain to preserve highlights, while applying bounded, colour-preserving shadow and midtone compensation before stretching. The default maximum display lift is two stops. It respects native bit depth, the camera’s gain/ISO steps and its ability to accept exposure commands.

The implementation evolved to include:

- Proportional response and faster recovery from severe clipping.
- Accounting for commands already in flight, avoiding repeated corrections from stale captures.
- Smooth engagement and release, plus recovery from gain limits.
- Separate day/night gamma overrides, where zero inherits ordinary gamma.
- Preservation of rendering state through worker reloads.
- Rejection of unsuitable or repaired purple-frame data for exposure decisions.
- An optional rendered-output guard, measuring the processed image before overlays and associating feedback with the exposure that produced it.
- Optional daytime blue/purple fringe reduction near bright edges, before sharpening.

Master protection, rendered-output protection and fringe reduction have separate switches. All three default off. Focus mode bypasses the feature; unsupported cameras cannot acquire its commanded-exposure behavior.

**Why this matters:** both acquisition and final appearance differ. A comparison of washed-out highlights, exposure oscillation, brightness, gamma or shadow noise may be influenced by local code. Compensation does not restore detail already lost through sensor clipping.

Sources: [highlight control](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/highlight.py), [metering](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/highlight_meter.py), [exposure controller](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/exposure/exposureBase.py).

### Twilight highlight prediction

**Added 19 October; prediction expanded 24 October and output feedback added 25 October. Still custom.**

The optional twilight extension measures a feathered outer horizon ring and uses capture-time Sun elevation. It forecasts brightening after accounting for changes made by its own exposure/gain controller.

Later additions cover whole-image brightening and separate red, green and blue tail trends. This helps react when clipping starts away from the horizon or in only one channel. Pending commands, unreliable clipped measurements and release speed are constrained.

A further optional mode follows the actual processed output tail, with a default target of 252/255. It allows a bounded amount of bright source pixels, with allowances related to Moon altitude and phase. Broad cloud or horizon clipping remains constrained.

Twilight protection and twilight output feedback default off and require master highlight protection. Their fixed solar-altitude safety fades are separate from the configurable smooth day/night transition.

**Why this matters:** sunrise and sunset exposure curves can differ despite identical target ADU. Record the twilight protection and output-feedback switches separately from the smooth-transition switch.

Sources: [twilight highlight policy](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/highlight_twilight.py), [output metering](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/highlight_output.py).

### Moon exception and Moon-night exposure preference

**Added 25–26 October. Still custom.**

Within master highlight protection, the Moon exception defaults enabled. It projects the captured Moon using the saved lens/VirtualSky geometry and exempts only a bounded, phase-dependent disk and halo. Invalid geometry or a below-horizon Moon retains ordinary protection.

At astronomical night, after measured Moon clipping establishes the policy, supported gain models can prefer longer shutter and lower gain while retaining the compensated shadow floor. Unsupported gain policies keep their ordinary behavior.

The final October change keeps the physical Moon exclusion continuous through the night/twilight handover. The night-specific exposure preference still fades out for twilight.

**Why this matters:** production’s exposure time, gain, noise and clipping on Moon-lit nights can differ from upstream and from earlier production versions.

Source: [Moon-night policy](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/highlight_night.py).

### Smooth day/night transition

**First integrated 3 October; single-pass processing on 4 October; configurable endpoints and MTF green-removal fading on 7 October. Still custom. Source default off.**

The feature blends supported settings along a smooth curve driven by the Sun’s elevation at the exposure midpoint. Existing configurations initially inherit the operational day/night threshold as the day endpoint; the night endpoint defaults to −12°. Independent saved endpoints support wider or narrower transitions.

It ramps target brightness, minimum exposure, fixed gain in the basic strategy, cadence, white balance, gamma, saturation and sharpening. Automatic white balance, stretching, grayscale and contrast contributions fade on the same captured frame. Different capture times are not blended together.

Each filter runs at most once. Denoise algorithms and strengths still use the nearer endpoint, with the exact midpoint choosing day. MTF green-removal can fade between its enabled setting and neutral MTF when the other endpoint is disabled. Other discrete algorithm combinations retain a switch.

Partial summer nights receive only the blend corresponding to the Sun’s actual altitude. Operational day/night grouping, cooling, binning and file-period boundaries remain discrete. The FITS viewer and processing preview retain their explicitly selected fixed profile.

“Use Night Color Settings” keeps its normal meaning: if enabled, night colour settings remain in use during the day. Separate day/night colour profiles require that setting to be off.

**Why this matters:** the binary “day” or “night” label and stored settings do not fully describe the effective twilight settings. Intermediate gains also require suitable dark coverage; the transition does not synthesize intermediate darks.

Sources: [feature guide](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/docs/wiki/Smooth-Day-Night-Transition.md), [twilight interpolation](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/twilight.py).

### Star-aware denoising and defect repair

**Integrated 8 October; extended through 17 October. Still custom. Selected as an optional denoise algorithm.**

Star-aware denoise treats luminance noise, chroma mottling and compact dark defects while protecting measured point sources, coherent trails and resolved edges. It is a separate algorithm alongside upstream’s existing denoise choices.

The subsequent work added:

- Bounded coordinate/confidence history to protect persistent faint sources and distinguish sensor-fixed impulses from moving stars.
- Isolated hot-pixel and carefully qualified cold-pixel-pair repair.
- Protection for resolved or nearly stationary stars such as Polaris.
- Cloud-aware checks on the quality and spatial consistency of apparent sky motion.
- Acquisition-time fading of star protection between Sun elevations of −6° and −8°.
- Image-circle bounds, ARM-oriented processing and reductions in repeated statistics, memory and copies.

History contains coordinates and confidence, not old image pixels. The algorithm does not blend earlier images into the current one. Bayer cold-defect handling works on a display copy before debayering and leaves saved FITS pixels unchanged.

Two independently disabled aids were added on 17 October:

- **VirtualSky assistance:** predicted catalogue positions confirm sources already measured in the current image. It checks alignment and geometry and is not a generated star replacement or a blanket star mask.
- **BPM assistance:** compatible active bad-pixel maps supply candidate coordinates. The code validates camera, depth, binning, dimensions, Bayer pattern and sensor origin, caches map coordinates and rechecks versions. It repairs only qualified defects visible in the current frame. This map selection is separate from normal calibration subtraction.

**Why this matters:** noise, small defects, star appearance and detail can differ substantially. A single archived-FITS preview lacks the live coordinate history. Local dark-patch repair also changes aperture/background flux, so display results are not a photometric reference.

Sources: [sky denoise](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/sky_denoise.py), [coordinate history](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/sky_catalogue.py), [VirtualSky positional reference](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/sky_reference.py).

### Higher colour precision and softer daytime saturation

**Integrated 8 October and 10 October. Still custom. No master feature switch.**

For non-focus, 16-bit colour input, production retains higher precision through geometry, green removal, white balance, saturation, gamma and sharpening. Detection uses a temporary 8-bit view, then rendering resumes from the retained high-depth pixels. Final display rounding to 8-bit follows sharpening, before 8-bit CLAHE and overlays. Depth-aware lookup tables and arithmetic avoid early quantization of smooth sky gradients.

With separate day colour settings and increased saturation, the daytime saturation shoulder approaches full saturation more gently. With smooth transitions enabled, that policy fades toward the night setting through twilight.

**Why this matters:** disabling highlight protection, smooth transitions and star-aware denoise still does not guarantee upstream-equivalent pixels. Banding, gradients, white balance, gamma, saturation and green-removal comparisons need an actual upstream render.

Sources: [image pipeline](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/image.py), [processing functions](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/processing.py), [SCNR](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/scnr.py).

## Capture lifecycle and performance

### Fresh exposure feedback despite delayed rendering

**Added 8 October. Still custom; active with master highlight protection outside focus mode.**

A supervised metering worker processes newly acquired frames ahead of expensive display rendering and forwards originals to the ordered renderer. Exposure commands use fresh capture information instead of waiting behind old display work.

Feedback is tied to its source capture, checked against camera/mode/profile and age, and discarded when stale or when workers stop. Optional processed-output control must establish fresh feedback again after interruption. Render backlog and spool usage are monitored; critical storage or RAM pressure can pause acquisition without discarding already captured files.

**Why this matters:** exposure response under rendering or video load differs from upstream, even when the displayed image visibly lags.

Sources: [metering worker](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/highlight_meter.py), [backlog tracking](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/render_backlog.py).

### Capture periods and orderly shutdown

**Added 8–9 October. Still custom, including with optional rendering features off.**

Capture mode, date and period are reserved before exposure and attached when the asynchronous frame arrives. End-of-period processing waits for expected frames to arrive and be saved, rather than relying only on the camera’s READY state.

Sequence and checkpoint state detect lost callbacks, queue tails and interrupted periods. Pause, reload and service stop drain final camera callbacks and image work in order. Passive cameras retain externally triggered delivery. Products can still be generated from available images after recorded gaps, with incomplete state preserved.

The service unit and unattended upgrade workflow were updated so this stop/drain behavior is deployed.

**Why this matters:** last-frame inclusion, date grouping, timelapse completeness, restart and shutdown behavior can differ from stock.

Source: [capture periods](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/capture_period.py).

### Capture watchdog

**Integrated 4 October. Still custom.**

The parent watches actual complete-frame arrival. Repeated exposure commands or READY responses do not count as progress.

A positive `CCD_EXPOSURE_TIMEOUT` remains a fixed timeout. **Zero selects adaptive timing**, based on exposure/cadence and recent processing allowance. The source default remains 330 seconds, so adaptive timing is available rather than automatically selected.

On expiry, the worker is asked to unwind; after a three-second grace, an unresponsive worker can be killed and replaced with fresh watchdog state. Expected pauses, disabled daytime capture, maintenance and lifecycle changes are handled explicitly.

**Why this matters:** production may recover from a stall that persists upstream. Recovery does not prove that the triggering camera fault never occurred.

Source: [capture watchdog](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/capture_watchdog.py).

### Lower batch priority and CPU reservation

**Priority work integrated 8 October; explicit CPU budget on 25 October. Still custom.**

The video worker lowers its own CPU and I/O priority after starting. `VIDEO_CPU_LIMIT` defaults to one CPU; zero disables explicit limiting. Supported systems constrain batch CPU affinity and ffmpeg thread use, leaving CPU capacity for capture and image rendering where possible.

**Why this matters:** generation speed, system load and camera responsiveness during video creation are not representative of unmodified upstream scheduling.

Sources: [video worker](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/video.py), [timelapse processing](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/timelapse.py).

### Background compressed FITS saving

**Integrated 25 October. Still custom.**

When compressed FITS saving is enabled and background capacity is available, production freezes an immutable HDU snapshot at the existing save point. The low-priority worker then performs lossless compression and publishes the database/upload record after success.

Pending snapshots survive worker restarts. Staging is bounded at eight snapshots; full or unavailable background capacity falls back to synchronous saving. Uncompressed FITS saves remain synchronous.

**Why this matters:** rendering latency and the time at which a FITS appears can differ, while the selected save point and image values are preserved.

Source: [FITS saving](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/fits_save.py).

## Synchronization and integrations

### Archive SyncAPI

**Introduced 13 September; expanded 14–15 September; consolidated as Archive sync on 9 October; TLS fix on 18 October. Still custom. PR #3267 open.**

Archive sync complements the existing Live sync mode with finite runs for intermittently available receivers.

It transfers retained local-camera media older than ten minutes, globally oldest first across selected types, including archives older than thirty days. Associated thumbnails and metadata are included. Standard images, panoramas, videos, mini-timelapses, keograms and startrail products are selected by default; FITS and RAW are optional. Hidden cameras and unfinished media are excluded.

The UI adds Sync now, cancellation, upload-speed selection, progress, byte/file/item counts, recent speed and next-action status. A service-owned availability schedule uses authenticated checks and an optional startup wait/recheck, and runs with the browser closed. Configuration preferences use the normal save/export/import workflow.

Persistent checkpoints and receiver size/SHA-256 lookups recover interrupted acknowledgements without duplicate upload. Dedicated POST lookups require Archive support on both ends. Temporary failures receive bounded retries; actionable account/certificate failures pause scheduling. The later TLS fix treats interrupted handshakes and EOF as retryable connections while retaining certificate/hostname validation failures as actionable errors.

Additional work hardened multipart integrity, metadata validation, receiver paths and temporary-file cleanup. Status rows remain stable during progress and recover after polling errors.

Cancelling a manual run preserves an already-enabled schedule; cancelling a scheduled run pauses it until re-enabled. Archive sync does not reconcile receiver changes or deletions and does not transfer configuration, so it is not a full mirror or backup.

Source defaults: Live mode; schedule off; checks every ten minutes; no startup delay; unlimited transfer speed.

**Why this matters:** archive behavior is installation-specific, and some hardening also changes the shared Live sender/receiver paths. This installation is not a stock SyncAPI reference.

Sources: [Archive guide](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/docs/wiki/SyncAPI-Archive.md), [transfer implementation](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/syncapi_sync.py), [scheduler](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/syncapi_schedule.py), [PR #3267](https://github.com/aaronwmorris/indi-allsky/pull/3267).

### Automatic timelapse completion events

**Integrated 9 October. Still custom.**

With MQTT enabled, automatic day/night batches publish a JSON completion event at `<base>/timelapse/complete`. Success checks task outcomes and expected local output files. It does not wait for remote uploads.

Payloads identify the camera, capture date, day/night period, batch, UTC completion time, output results and failures. Manual regenerations and mini-timelapses are excluded. Completion events are non-retained; ordinary broker/queue delivery behavior still applies.

Home Assistant discovery adds a Timelapse completed event entity with success/failed event types. Discovery must be rerun after adding the feature.

**Why this matters:** these completion triggers are local functionality. They are separate from the older Home Assistant control API experiment that was removed.

Sources: [completion handling](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/timelapse_completion.py), [MQTT guide](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/docs/wiki/MQTT-Broker-Publishing.md).

## Web interface and exports

### Custom time-range keograms

**Integrated 4 October. Still custom.**

An authorized user can select a local camera and arbitrary start/end times, including ranges crossing normal day/night boundaries. The existing background worker processes retained, non-excluded images chronologically using current keogram settings.

The tool handles up to 20,000 candidates, reports used/skipped/resized images and actual timestamps, skips missing or unreadable originals and tolerates compatible resolution changes. It exports a separate JPEG.

The last successful preview is retained per camera for 24 hours, restored in a fresh tab and downloadable with the selected range in the filename. Remote-camera and cloud-only source retrieval is not supplied by this feature.

**Why this matters:** this export interface is custom; the source images also retain any other production rendering differences.

Source: [custom keogram implementation](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/customKeogram.py).

### Gallery panorama switch

**Integrated 27–28 September. Still custom. PR #3225 open.**

The existing gallery viewer can switch the selected capture between normal image and matching saved panorama. Panorama navigation skips captures without panoramas.

Image, Download, Loop and Mini-Timelapse actions follow the displayed source. Exclusion remains associated with the original capture. Keyboard handling avoids duplicate handlers or conflicting page navigation. The toolbar wraps and aligns controls on smaller or scaled displays, with viewer padding following toolbar height.

This replaced the initial separate panorama link. It changes navigation and presentation, not panorama generation or pixels.

Source: [gallery template](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/flask/templates/gallery.html), [PR #3225](https://github.com/aaronwmorris/indi-allsky/pull/3225).

### VirtualSky mirroring and planet behavior

**Integrated 24 September, following development on 21–24 September. Still custom. PR #3201 open.**

The original tilt, pointing recovery and lens-calibration work is now upstream. Remaining production extensions include:

- Independent horizontal/vertical overlay mirroring, default off, preserving readable labels and leaving captured image pixels unchanged.
- Automatic solving of handedness alongside pointing and tilt, with rejection of ambiguous fits.
- Orientation-bound learned corrections, invalidated when geometry or flips change.
- Correct inverse projection and pointer lookup under mirroring.
- Planet interpolation across the right-ascension wrap, removal of the old Jupiter coordinate adjustment, and bounded interpolation intervals.
- Correction of overlapping Milky Way data reload state and script versioning.

The overlay controls alone do not mirror the camera image. Optional catalogue denoise assistance and Moon geometry can consume related alignment information.

Sources: [VirtualSky renderer](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/flask/static/virtualsky/virtualsky.js), [mirror-aware solver](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/lens_solver/solver.py), [PR #3201](https://github.com/aaronwmorris/indi-allsky/pull/3201).

### Latest capture-health warnings

**Integrated 24 August. Still custom.**

The normal Latest pages add warnings for unexpected worker death, stale successful images and camera/INDI errors, linking to the capture log. Expected silence during pause, disabled daytime capture and maintenance is suppressed. Remote and non-active cameras are handled separately.

Freshness uses actual output timing or a recent database image, with thresholds based on exposure/cadence and bounded health polling even during long exposures. The feature does not itself establish the root cause of a warning or change image pixels.

Source: [capture health](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/capture_health.py).

### DST freshness and VirtualSky time correction

**Freshness fix integrated 25 October; shared VirtualSky correction on 28 October. Still custom. PR #3309 open.**

Local cameras use the host’s current time directly instead of applying a stale saved UTC offset. Remote cameras retain offset conversion. VirtualSky reuses that shared calculation.

This prevents false local freshness windows and shifted overlay time across daylight-saving changes. It does not change physical exposure or the sensor’s captured pixels.

The final production commit also repairs the VirtualSky test fixture’s package context. That is test maintenance, not another user-facing feature.

Source: [shared view timing](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/flask/base_views.py), [PR #3309](https://github.com/aaronwmorris/indi-allsky/pull/3309).

### Charts and local build hygiene

**Chart fix integrated 19 October. Still custom. PR #3298 open.**

A small CSS sizing change prevents the chart/container resize feedback observed at fractional zoom on affected mobile Chrome configurations. It does not alter plotted measurements or image processing.

Production also excludes local test environments, scratch directories and development dependencies from Docker build context. These are build-hygiene changes. The separate unattended INDI notice fix is not part of production.

Source: [chart template](https://github.com/321MathiasS123/indi-allsky/blob/29ff82407661493fb0ffd467d8d178565cb13aed/indi_allsky/flask/templates/chart.html), [PR #3298](https://github.com/aaronwmorris/indi-allsky/pull/3298).

## Contributions now in upstream main

Dates in this table are GitHub merge timestamps converted to Europe/Berlin. They can be a calendar day later than the US date shown by a Git log.

| Merged | PR | Functionality | What reached upstream |
| --- | --- | --- | --- |
| 2026-08-14 | [#3019](https://github.com/aaronwmorris/indi-allsky/pull/3019) | ASI676MC purple-frame handling | Opt-in detection, exclusion/repair, calibration and diagnostic FITS; inherited before production began. |
| 2026-08-16 | [#3048](https://github.com/aaronwmorris/indi-allsky/pull/3048) | Panorama image exclusions | Panorama loops/timelapses honor exclusion of the source image; inherited before production began. |
| 2026-08-23 | [#3077](https://github.com/aaronwmorris/indi-allsky/pull/3077) | Gamma LUT mode refresh | Changing day/night gamma refreshes cached lookup tables. |
| 2026-08-24 | [#3065](https://github.com/aaronwmorris/indi-allsky/pull/3065) | Mini-timelapse overhaul | Normal/panorama sources, seam-aware static crop or panning, framing, previews, preflight and generation workflow. |
| 2026-08-25 | [#3086](https://github.com/aaronwmorris/indi-allsky/pull/3086) | Cardinal direction normalization | Wrapped rotation angles give correct image and panorama direction labels. |
| 2026-08-25 | [#3088](https://github.com/aaronwmorris/indi-allsky/pull/3088) | ASI676MC reserve validation | Marginal calibration groups can be replaced by reserve evidence. |
| 2026-08-28 | [#3104](https://github.com/aaronwmorris/indi-allsky/pull/3104) | Mini-timelapse bitrate | Selectable bitrate, size estimates and resolution/FPS guidance. |
| 2026-08-28 | [#3098](https://github.com/aaronwmorris/indi-allsky/pull/3098) | Process FITS detection dimensions | Detection masks/overlays remain consistent after processing changes dimensions. |
| 2026-09-18 | [#3143](https://github.com/aaronwmorris/indi-allsky/pull/3143) | Mini-Timelapses wiki | Documentation for panorama, preview and generation controls. |
| 2026-09-18 | [#3142](https://github.com/aaronwmorris/indi-allsky/pull/3142) | ASI676MC wiki | Integrated purple-frame repair and calibration guidance. |
| 2026-09-18 | [#3121](https://github.com/aaronwmorris/indi-allsky/pull/3121) | Loop and preview playback | Bounded lookahead, decoding/failure handling, cancellation, responsive timespan changes and mini-preview ordering. |
| 2026-09-18 | [#3129](https://github.com/aaronwmorris/indi-allsky/pull/3129) | VirtualSky tilt and calibration | Tilted projection, automatic pointing recovery, lens curvature, optional learned correction and calibration UI. |
| 2026-09-18 | [#3183](https://github.com/aaronwmorris/indi-allsky/pull/3183) | Allsky Map coordinate persistence | TN-1’s coordinate save/validation correction; included alongside the local work. |
| 2026-09-25 | [#3218](https://github.com/aaronwmorris/indi-allsky/pull/3218) | Notifications on already-open pages | Show newly created notifications on already-open pages. |
| 2026-09-25 | [#3217](https://github.com/aaronwmorris/indi-allsky/pull/3217) | Watchdog notification ordering | Create the expired-watchdog notification before returning DOWN status. |
| 2026-09-29 | [#3216](https://github.com/aaronwmorris/indi-allsky/pull/3216) | MTF LUT depth refresh | Rebuild the lookup table when image bit depth changes. |
| 2026-10-01 | [#3230](https://github.com/aaronwmorris/indi-allsky/pull/3230) | Notification response validation | Ignore HTML/malformed responses instead of opening an empty dialog; associated status polling recovery is also upstream. |
| 2026-10-05 | [#3253](https://github.com/aaronwmorris/indi-allsky/pull/3253) | Daytime MTF white balance | Use daytime channel settings when separate daytime colour is selected. |
| 2026-10-05 | [#3252](https://github.com/aaronwmorris/indi-allsky/pull/3252) | Sensor day/night settings | Correct SI7021 night heater selection and SI1145 daytime range logging. |
| 2026-10-12 | [#3268](https://github.com/aaronwmorris/indi-allsky/pull/3268) | ASI676MC archive recovery | Search retained reserves more widely, prefer complete reference triplets, retain strict acceptance and improve reporting. |

### What can now be treated as shared functionality

The mini-timelapse work includes panorama framing across the horizontal seam, fixed crops or movement between endpoints, aspect presets, direction/full-turn choices, duration/range preflight, previews, 30/60 FPS choices, background generation and the normal listing/download/delete workflow. Bitrate selection and size guidance followed. Later playback and documentation work is also upstream.

This does **not** mean mini-timelapses can be generated solely from remote or S3-only originals. The checked implementation requires usable local source images. The reviewed external PR [#3170](https://github.com/aaronwmorris/indi-allsky/pull/3170) was closed without merging.

VirtualSky’s original tilted-camera projection, automatic pointing recovery, lens curvature and optional additional calibration are upstream. The custom mirroring, orientation-dependent model handling, planet fixes and DST changes described above remain relevant when reproducing overlay issues.

The checked ASI676MC implementation and its two feature wiki pages have **no production-versus-upstream residual**. Original purple handling predated production; reserve replacement and full archive recovery were added later and upstreamed. Archive recovery prefers complete normal/purple/normal groups but can retain suitable pairs when needed. Original FITS are preserved, acceptance checks remain strict, and manual uploads retain their fixed evidence collection.

Individual gamma, MTF and daytime white-balance fixes are now shared. Production still changes the wider processing pipeline, so “the fix is upstream” does not make all local image output representative of main.

### Open PRs and closed alternatives

Six open upstream PRs correspond to functionality currently present in production: [dark library #3112](https://github.com/aaronwmorris/indi-allsky/pull/3112), [VirtualSky mirroring #3201](https://github.com/aaronwmorris/indi-allsky/pull/3201), [gallery panorama #3225](https://github.com/aaronwmorris/indi-allsky/pull/3225), [Archive sync #3267](https://github.com/aaronwmorris/indi-allsky/pull/3267), [chart sizing #3298](https://github.com/aaronwmorris/indi-allsky/pull/3298) and [DST timing #3309](https://github.com/aaronwmorris/indi-allsky/pull/3309).

The earlier dark-library PR [#3106](https://github.com/aaronwmorris/indi-allsky/pull/3106) was closed without merging; #3112 is the active replacement. The early panorama mini-timelapse PR [#3052](https://github.com/aaronwmorris/indi-allsky/pull/3052) was also closed without merging, while the broader implementation reached main through #3065. A closed PR must therefore not automatically be read as either “merged” or “functionality absent.”

## Functionality timeline

These are production integration dates, grouping repeated refinements into functional milestones.

| Period | Added or changed in production | Subsequent status |
| --- | --- | --- |
| Before 22 August | ASI676MC purple-frame handling and panorama exclusion propagation already inherited | Upstream before this branch’s surviving custom history |
| 22–24 August | Mini-timelapse overhaul restored; guided dark automation, profiles, temperature policy, library eligibility and maintenance; gamma fix | Mini/gamma later upstream; dark system remains custom |
| 24–26 August | Cardinal directions, ASI reserve validation, Latest capture-health warnings, Process FITS detection geometry | All except capture-health warning later upstream |
| 28 August | Mini-timelapse bitrate selection, estimates and guidance | Upstream |
| 30 August–3 September | Consolidated dark UI, sensorless matching, deletion/timeout fixes and exposure-module integration | Custom dark functionality remains |
| 5–10 September | Loop buffering/cancellation; mini-preview ordering; VirtualSky tilt, pointing, lens correction, masks, UI and cache work | Main functionality later upstream |
| 13 September | On-demand incremental SyncAPI, retry diagnostics and availability schedule | Evolved into current custom Archive sync |
| 13 September | Home Assistant control/recovery API briefly integrated and explicitly reverted | Removed |
| 14–15 September | Sync speed limit, multipart fix, progress/rates and stable status/polling | Custom Archive work |
| 17–19 September | Mini-crop validation safeguard, map coordinate persistence, consolidated dark cooldown/docs | Mini/map shared upstream; dark cooldown custom |
| 24–29 September | VirtualSky mirroring/planet corrections; MTF cache fix; notifications; gallery panorama and toolbar | Mirroring/gallery custom; MTF/notification fixes upstream |
| 1 October | ASI retained-archive recovery and clearer calibration reporting | Later upstream, including 10 October refinements |
| 2–5 October | Highlight exposure/rendering control and early tuning; daytime MTF/sensor corrections; smooth transitions; custom keogram; actual-frame watchdog | MTF/sensor corrections upstream; larger features custom |
| 7 October | Independent twilight endpoints, save validation and MTF SCNR fading | Custom |
| 8 October | Star-aware denoise, colour precision, fresh metering, capture-period barriers, worker priority and defect/source protection | Custom |
| 9 October | Capture drain refinements, MQTT batch completion and HA discovery; Archive naming, configuration and recovery consolidation | Custom |
| 10–11 October | ASI complete-group preference; soft daytime saturation; highlight fringe/reload/response/recovery work; denoise continuity | ASI recovery upstream; rendering/controller changes custom |
| 16–18 October | Polaris protection, VirtualSky-assisted denoise, cached BPMs, cloud-aware motion and SyncAPI TLS classification | Custom |
| 19 October | Chart zoom sizing and twilight horizon highlight protection | Custom |
| 24 October | Whole-image and per-colour twilight forecasts | Custom |
| 25–26 October | DST freshness, output-driven twilight control, live CPU budget, background compressed FITS, Moon-night exception/gain policy and continuous handover | Custom |
| 28 October | VirtualSky shared DST offset and fixture correction | Runtime correction custom; fixture is test maintenance |

## Experiments and changes that are not current production features

### Temperature-controlled dark cooldown

`feature/dark-temperature-cooldown` contains additional temperature-based waits, temperature-check exposures and cooled-camera planning/compatibility work. Its relevant implementation commits are not production ancestors and `dark_temperature.py` is absent from production. Production has the fixed/exposure-matched cooldown described earlier.

### Alternative night-sky renderer

The separate `feature/night-sky-pipeline` experiment is not integrated. Its night-sky renderer modules and lab pipeline are absent from the production tree. It must not be confused with the active star-aware denoiser or colour-precision work.

### Home Assistant control and recovery API

The authenticated control API, coordinated recovery/reboot integration, supporting services and example configuration were tried on 13 September and explicitly removed by [`b9955314`](https://github.com/321MathiasS123/indi-allsky/commit/b995531400c49bb2e44cbc6a7326834aad269b63). The current MQTT completion event is a separate later feature.

### Unattended INDI build notice

The one-line `BUILD_INDI_SETTINGS` notice fix, [PR #3306](https://github.com/aaronwmorris/indi-allsky/pull/3306), is open but its feature commit is **not in production**. It should be tracked as prepared work outside this production inventory.

### Superseded interfaces and branch histories

The original separate gallery panorama link was removed and replaced by the current viewer switch. The temporary mini-timelapse reverts were undone in production, and the feature later reached upstream.

Retained feature branches are not a reliable substitute for checking the production tree. For example, current dark-library planning matches production, but the retained branch also has an explicit gain-snapping line in dark capture that production lacks. A branch name or a recorded integration merge cannot establish that every later branch change is present.

## Using this installation when responding to upstream issues

### Differences that matter most

| Issue area | What to account for before describing a local result as an upstream reproduction |
| --- | --- |
| Calibration, colour cast or fixed-pattern noise | Selected dark/BPM IDs, gain/exposure/temperature, active sets and production’s nearest-temperature policy |
| Banding, gamma, white balance, saturation or SCNR | Higher-precision pipeline and daytime saturation shoulder, even with optional features off |
| Clipping, auto exposure, sunrise/sunset or Moon-night brightness | Highlight master/output/twilight/Moon settings, fresh metering and actual command history |
| Hot/cold pixels, faint stars, trails or noise reduction | Star-aware algorithm, strength, catalogue/BPM switches and live temporal history |
| Day/night jumps and timelapse flicker | Smooth-transition endpoints, capture midpoint, effective gain/target/profile and discrete settings |
| Camera stalls, restart or missing final frames | Actual-frame watchdog, fixed/adaptive timeout, period drain and fresh-worker recovery |
| Load-dependent faults | CPU budget/priority, compressed FITS background saving and backlog/spool pressure |
| SyncAPI transfers | Live versus Archive mode, both receiver versions, hardened lookup/retry/TLS behavior |
| UI, overlay or freshness | Custom gallery, mirroring/planet corrections, capture-health warnings, DST fix and chart CSS |

A local observation can still be useful. Describe it as a result on the modified production branch until the relevant behavior has been compared with upstream at a stated revision. A matching result on upstream is stronger evidence than assuming that disabling the obvious feature switches removed every difference.

### What a saved FITS does and does not reproduce

| Evidence | What it contains or tests |
| --- | --- |
| Untouched diagnostic camera FITS | Strongest evidence for original camera output, including the purple-frame problem |
| Standard pre-dark FITS | Before dark subtraction, but it may already contain ASI purple-frame correction |
| Standard post-dark FITS | May already contain dark/BPM calibration and early hole repair |
| Rendering the same saved FITS on two revisions | Tests processing after that saved state; does not test original camera commands or master selection |
| One offline frame with star-aware denoise | Does not recreate live source/defect history |
| Finished JPEG, panorama or timelapse | Contains the combined acquisition, calibration and rendering effects that produced its source images |

Display denoising, highlight shadow lift and colour/gamma precision do not directly rewrite the standard FITS HDU. Highlight control still affects future FITS indirectly by changing the exposure and gain used to acquire them.

For image-processing investigations, retain the untouched/pre-dark evidence when available, selected masters, effective settings and actual revision. Do not subtract darks again from an already calibrated FITS simply to imitate the acquisition path.

## Maintaining this inventory

Update this document whenever production gains, changes, removes or reverts functionality, and whenever an upstream merge changes a recorded feature's status. Include the update in the production integration commit or an accompanying documentation commit. Keep the inventory on production so installation-specific history stays out of isolated upstream feature PRs.

1. Refresh origin and upstream, then record the reviewed production code revision, upstream main revision and their merge base. Keep the functional snapshot separate from later documentation-only commits; the document does not need to name the commit that contains its own update.
2. Compare the actual production and upstream trees and review production's first-parent integration history. Use the full history for feature development details. Separate custom differences from upstream work production has not yet incorporated.
3. Check the linked PRs' current merged status and what functionality actually reached main. A closed PR or changed commit hash alone is insufficient. Keep any remaining local extensions marked custom, even when the base feature is upstream.
4. Update the status table, affected descriptions, functionality timeline and issue-reproduction guidance. Preserve introduction and upstream merge dates. Record removals and separate experiments without presenting them as active production features.
5. Refresh the review date, revisions, snapshot counts and affected source links for functional reviews. Use immutable GitHub links for implementation evidence and Europe/Berlin dates for upstream merge timestamps. Keep source defaults distinct from verified live settings and deployment provenance.

The initial review used the production tree and integration history, comparisons with fetched upstream main and live PR metadata. Source permalinks throughout the feature descriptions preserve its implementation evidence. The recorded baseline can also be compared directly through the [custom production delta](https://github.com/321MathiasS123/indi-allsky/compare/18b8407d31843f3df9c69f8b5b16443f7df65c82...29ff82407661493fb0ffd467d8d178565cb13aed).

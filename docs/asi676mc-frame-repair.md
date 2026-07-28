# ASI676MC frame repair and diagnostic FITS

This document records the custom ASI676MC feature carried by
`dev/asi676mc-image-correction`. It is intended to give a future maintainer—or
another Codex chat—enough context to modify or remove the feature without
disturbing the rest of the indi-allsky pipeline.

## Change history

The feature was developed as the following commit series on top of upstream
commit `3d834018`:

| Commit | Purpose |
| --- | --- |
| `f9096757` | Add optional ASI676MC RAW16 frame correction |
| `7e965b26` | Add repair logging and gallery audit metadata |
| `684ad221` | Refine controls, camera gating, and timing |
| `0912d47c` | Reduce CPU and memory use on older Raspberry Pis |
| `4b7064b4` | Move the settings group to the bottom of the Image tab |
| `4e553308` | Save paired diagnostic FITS and add download controls |
| `35e2969f` | Add this maintenance and removal guide |
| `9909cc35` | Fix Image Viewer initialization with diagnostic downloads |

Use `git show <commit>` for the exact historical patch. When the branch has
continued to evolve, use these commits as a map rather than blindly reverting
them.

## Runtime behavior

The repair is opt-in through `IMAGE_ASI676MC_REPAIR.ENABLE` and is gated by the
detected camera name. It only runs for an ASI676MC with supported RAW16 RGGB,
1x1-binned input. Detection and repair happen immediately after the input FITS
is opened and before calibration, debayering, stacking, or later image
processing.

The repair implementation is in `indi_allsky/asi676mc.py`; pipeline integration
is in `indi_allsky/processing.py` and `indi_allsky/image.py`.

### Saturated-highlight reconstruction

The camera fault also clips both source green samples before their inverse gain
correction. Without a second reconstruction step, the repaired green planes
plateau below the corrected red and blue planes, leaving saturated highlights
magenta even though unsaturated areas are repaired correctly.

`_pack_clipped_green_masks()` records two compact masks after row-phase
restoration and before applying the gain lookup tables:

- source G1 samples at or above `SOURCE_SATURATION_THRESHOLD`, used by the
  original neighboring-G2 interpolation;
- cells where both source green samples are clipped.

After gain correction and the original G1 interpolation,
`_reconstruct_clipped_green()` handles only the jointly clipped cells. Because
their true green values are no longer recoverable, it raises both corrected
green values to at least the larger corrected red/blue value in the same Bayer
cell. This neutralizes false magenta highlights while leaving all recoverable
samples and all normal frames untouched.

The joint mask is bit-packed and adds 394,272 bytes at the ASI676MC's
3552-by-3552 resolution. Reconstruction remains chunked. It is only reached
after a frame has been classified as bad, so normal-frame detection cost is
unchanged.

### Diagnostic FITS pair capture

Commit `4e553308` added the independently selectable
`IMAGE_ASI676MC_REPAIR.SAVE_DIAGNOSTIC_FITS` option. When enabled:

1. A frame classified as `repaired` or `validation_failed` causes the untouched
   incoming FITS file to be copied before the source is removed.
2. The immediately following successfully ingested frame for the same camera is
   copied as the comparison FITS.
3. Consecutive bad frames share one physical file where appropriate: a frame
   can be both the `following` member of one pair and the `bad` member of the
   next pair.
4. Copying and database/upload failures are logged and isolated so diagnostic
   capture cannot stop the normal image pipeline.

Pair state is held in `ImageWorker.asi676mc_diagnostic_pending`, keyed by camera
ID. A service restart between the bad frame and its successor can therefore
leave a pair with only its bad member.

The copies are standard `IndiAllSkyDbFitsImageTable` assets. Existing FITS
expiration, local URL, file-transfer, and S3 behavior applies. No database
migration was added.

The FITS row JSON uses this shape:

```json
{
  "asi676mc_diagnostic": {
    "version": 1,
    "source": "untouched_input",
    "roles": [
      {
        "capture_id": "<pair UUID>",
        "role": "bad"
      }
    ]
  }
}
```

The associated rendered image row stores a small
`asi676mc_diagnostic_fits` JSON object containing the FITS row ID and the same
role list. Pair UUIDs, rather than timestamps, associate the bad and following
files.

Downloads appear as `Bad FITS` and `Next FITS` in the standard Image Viewer
download strip. The enlarged gallery viewer also exposes the pair, but gallery
thumbnails deliberately do not contain download badges.

## Files involved in diagnostic capture

- `indi_allsky/asi676mc.py`
  - `DIAGNOSTIC_METADATA_KEY`
  - `DIAGNOSTIC_BAD_STATUSES`
  - `diagnostic_capture_plan()`
- `indi_allsky/image.py`
  - per-camera pending-pair state
  - call immediately after `correct_asi676mc_frame()`
  - `capture_asi676mc_diagnostic_fits()`
  - `_archive_asi676mc_diagnostic_fits()`
  - image-row diagnostic metadata
- `indi_allsky/config.py`
  - default for `SAVE_DIAGNOSTIC_FITS`
- `indi_allsky/flask/forms.py`
  - settings field
  - diagnostic FITS lookup/pairing
  - Image Viewer and gallery JSON fields
  - labels in the standard FITS viewer
- `indi_allsky/flask/views.py`
  - settings load/save wiring
  - camera-gated Image Viewer and gallery wiring
- `indi_allsky/flask/templates/config.html`
  - switch, help text, submission list, and master-switch grouping
- `indi_allsky/flask/templates/imageviewer.html`
  - `Bad FITS` and `Next FITS` controls
- `indi_allsky/flask/templates/gallery.html`
  - enlarged-view FITS controls and slide data attributes
- `testing/image/test_asi676mc_repair.py`
  - bad/following and consecutive-bad pairing tests

## Removing only diagnostic FITS capture

This is the preferred removal scope if frame correction itself is still useful.

1. Remove `SAVE_DIAGNOSTIC_FITS` from the default configuration, settings form,
   settings template, and settings load/save code.
2. Remove the diagnostic constants and `diagnostic_capture_plan()` from
   `indi_allsky/asi676mc.py`.
3. From `ImageWorker`, remove the pending-pair dictionary, the guarded
   `capture_asi676mc_diagnostic_fits()` call, both diagnostic capture methods,
   and persistence of `asi676mc_diagnostic_fits` into image metadata.
4. Remove `_asi676mc_diagnostic_assets()` and its Image Viewer/gallery response
   fields from `indi_allsky/flask/forms.py`. Remove the diagnostic labels from
   the FITS viewer. The camera-specific filter added to the ordinary
   same-timestamp FITS lookup is safe to retain.
5. Remove the Image Viewer diagnostic spans and JavaScript.
6. Remove only the diagnostic FITS data attributes, label CSS, and PhotoSwipe
   registration helper from the gallery template. Keep repair outlines, repair
   badges, and the repaired-only filter.
7. Remove the diagnostic pairing tests, but retain all detection, repair,
   validation, timing, and memory tests.
8. Run the ASI676MC unit tests and JavaScript syntax checks for the config,
   gallery, and Image Viewer templates.

Removing this code needs no database migration. Existing diagnostic FITS remain
valid ordinary FITS records and will expire according to
`IMAGE_FITS_EXPIRE_DAYS`. Existing image/FITS JSON keys are harmless if left in
the database. Prefer normal expiration over manually deleting files and rows.

## Removing the complete ASI676MC feature

If the camera issue is fixed upstream or this customization is no longer
wanted, first follow the diagnostic-removal steps above, then:

1. Remove `indi_allsky/asi676mc.py` and its test module.
2. Remove the ASI676MC result property from the processing image-reference
   object.
3. Remove `correct_asi676mc_frame()`, its helper, and the early pipeline call.
4. Remove the complete `IMAGE_ASI676MC_REPAIR` configuration block and all
   corresponding form validators, fields, view wiring, camera-support checks,
   and Image-tab controls.
5. Remove repair metadata persistence and all gallery repair status, outline,
   badge, tooltip, and filter code.
6. Search the tree for both `asi676mc` and `IMAGE_ASI676MC_REPAIR`; no runtime
   references should remain.
7. Compare the final result against the commit series above and run the
   relevant image and web-template checks.

Stored JSON audit data does not require cleanup when the complete feature is
removed.

## Removing only saturated-highlight reconstruction

To retain the original row-phase and gain repair but remove the later
highlight refinement:

1. Replace the runtime call to `_pack_clipped_green_masks()` with
   `_pack_clipped_green_mask()` and keep only `green1_clipped_packed`.
2. Remove `both_green_clipped_packed` from `_reconstruct_clipped_green()` and
   delete the block beginning with its `numpy.unpackbits()` call.
3. Remove `_pack_clipped_green_masks()`, restoring the original single-mask
   implementation inside `_pack_clipped_green_mask()`.
4. Remove the jointly-clipped-green unit test and the joint-mask assertions
   from the partial-byte test.

This narrower removal does not affect configuration, database rows, gallery
metadata, or diagnostic FITS capture.

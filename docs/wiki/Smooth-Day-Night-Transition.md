# Smooth day/night transition

Enable **Smooth day/night transition** in its own **Location → Smooth Day/Night
Transition** box. It is disabled by default.

Set **Full day settings at Sun elevation** and **Full night settings at Sun
elevation** independently. The night endpoint must be lower; both accept −90°
to +90°. Evening blends from the day endpoint to the night endpoint; morning
follows the same smooth S-shaped curve backwards. For example, 0° to −12° gives
a wider interval than −3° to −9°. These endpoints also control ordinary gamma
and any installed highlight-protection gamma overrides.

The operational day/night **Sun altitude** stays separate. Existing configurations
continue using that altitude as the day endpoint until a separate value is saved.
The usual −6°/−12° interval therefore stays unchanged on update. The form shows
the inherited endpoint; saving it makes that value independent of the mode threshold.

The blend follows solar elevation, so its duration changes with location and
season. If a summer night only reaches part of the interval, the settings stop
at that partial blend and return towards day. They are never stretched to force
full night settings. The configuration page shows the duration, or maximum
partial blend, for the current solar cycle using the saved location and settings.
Save changes and reload the page to refresh this estimate. UTC labels identify
the predicted reversal time on partial nights. Near the poles, no reversal time
is shown if the Sun keeps moving in one direction through the solar cycle.

## Settings covered

- Target ADU, with small exposure adjustments following the changing target even
  inside the normal brightness tolerance. Outside that tolerance, ordinary
  exposure correction applies. Highlight protection keeps its clipping policy.
- Minimum exposure, interpolated logarithmically between the camera-resolved
  day/night limits, rounded to the existing microsecond precision and capped by
  the camera maximum. Exposure remains controlled by measured image brightness.
- Fixed gain in the basic exposure strategy. Automatic-gain strategies retain
  their existing gain policy and receive the changing target/minimum exposure.
- Capture interval, excluding dedicated SQM exposures and focus mode.
- Manual white balance, midtone white balance, gamma, saturation and sharpening.
- Automatic white balance, stretching, grayscale and contrast enhancement.
  Effects are faded using the same captured frame; no images from different
  capture times are blended.
- Denoising runs once with the nearer endpoint's algorithm, strength and options.
  Green removal also runs once with the nearer algorithm; its continuous midtone
  setting is blended. Above 50% night settings the night profile is selected;
  at or below 50% the day profile is selected. With −6°/−12° endpoints, this
  switches at −9°: night denoising during the first half of the morning blend,
  then day denoising. Evening reverses this selection. Partial summer nights
  that never reach 50% night settings retain the day algorithms.

**Use Night Color Settings** keeps its existing meaning: when enabled, those
color settings remain the night settings throughout the day. Turn it off to
transition between separately configured day and night color profiles.

Image processing uses each exposure's midpoint, derived from its receipt time,
elapsed capture/readout time and exposure length. Queued images do not use the
Sun's position at their later processing time. Runtime values do not overwrite
saved settings. Logs report the solar elevation and percentage of night settings.
Timestamp arithmetic uses UTC, including across daylight-saving changes. With
no recorded elapsed time, the midpoint is estimated as half an exposure before
the receipt time.

The FITS viewer and image-processing preview keep their explicitly selected
fixed settings; automatic twilight blending applies to the capture pipeline.

## Highlight protection

This feature does not require highlight protection. If the separate
highlight-protection feature is installed, both can be enabled. Its exposure target and shadow
compensation receive the same blended target ADU. Day/night highlight-gamma
overrides are resolved first (zero means inherit the corresponding ordinary
gamma), then interpolated. Highlight clipping limits and maximum shadow lift
retain their existing meaning. Invalid frames, and repaired frames excluded by
highlight protection, continue to hold exposure and gain.

## Limits and validation

Operational day/night classification remains at the existing Sun altitude:
daytime capture policy, file grouping, timelapse boundaries and cooling retain
their behavior. Binning, camera formats/readout properties, stacking policies,
Moon Mode changes and the legacy automatic-gain steps are discrete. Matching
day/night binning and formats avoids those additional image discontinuities.

Intermediate fixed gains use the existing dark-selection policy. A library
containing only the two endpoint gains can select higher-gain darks during
twilight; verify the matched dark frames and provide suitable calibration
coverage before relying on the ramp. Calibration is not synthesized or disabled.

Each filter runs at most once per frame. Different discrete algorithms or
denoising strengths can produce a visible change at the midpoint; using matching
day/night settings avoids that change.
Automatic exposure still reacts to clouds and changing illumination; enabling
the transition does not guarantee a flicker-free video under every condition.

Validate a real dusk and dawn sequence with the intended camera, dark library,
gain strategy and processing profiles before production use. Disabling the
feature restores the existing day/night selection and exposure behavior.

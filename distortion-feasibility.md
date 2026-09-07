# VirtualSky distortion calibration: feasibility assessment

**Star-based lens calibration is broadly applicable, including rectangular
sensors and cropped image circles. Adding one or two radial terms to the
current equisolid projection is useful, but is not a universal solution.**

This local experiment is isolated on `experiment/virtualsky-distortion-feasibility`,
based on tilt commit `63084b5f249dbb014a8ab73763a2cdf4d1a62591`.
It changes no application code, configuration, production branch, or open PR.

The reproducible study is `tests/lens_solver/test_distortion_feasibility.py`;
raw measurements are in `distortion-feasibility-results.json`. Its tests are
discovered by pytest. All simulated star identities are supplied, with no false
matches. These results assess calibration after matching, not blind solving.

## What generalizes

Calibration maps a star's direction relative to the **lens axis** to an image
position. Camera tilt, heading, and image rotation do not change the lens's
radial law. Distortion must therefore be calculated about the optical axis,
not the geographic zenith or necessarily the centre of the image rectangle.

A rectangular sensor or ordinary crop only limits which points are visible;
there is no need to detect the rim or see the entire image circle. The existing
diameter is the equisolid model's 180-degree reference scale, not necessarily a
visible circle or the lens's rated field of view. A future model must keep its
valid angular range separate from the sensor dimensions and this legacy scale.

Known camera calibration models cover conventional, wide-angle, and fisheye
lenses using a flexible angle-to-radius curve. Their extensions also account
for asymmetric distortion and differing horizontal/vertical pixel scales.
[Kannala and Brandt, 2006](https://users.aalto.fi/~kannalj1/calibration/Kannala_Brandt_calibration.pdf).
OpenCV similarly exposes several fisheye coefficients and options to fix
parameters or check conditioning.
[OpenCV fisheye documentation](https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html).

## Measurements

The first experiment fits scale plus zero, one, or two radial terms to
independently generated ideal lens projections. The proposed correction is
`r = a0*q + a1*q^3 + a2*q^5`, with `q = sin(theta/2)/sin(45 degrees)`.
Zero correction terms reproduce the existing equisolid projection.
Each lens's outer radius is 1,000 pixels; FOV is the angular diameter of the
sampled circular field, not a manufacturer's horizontal or diagonal rating.
There are 1,000 calibration rays and 10,000 independent validation rays,
uniformly distributed in solid angle. Orientation and centre are known here.
Errors measure model approximation without detection noise.

| Ideal lens / FOV | Current curve RMS | Two-term curve RMS | Two-term largest edge error |
| --- | ---: | ---: | ---: |
| Equisolid / 180 degrees | 0 px | 0 px | 0 px |
| Equidistant / 180 degrees | 17.51 px | 0.13 px | 0.42 px |
| Orthographic / 180 degrees | 71.28 px | 0.27 px | 0.85 px |
| Stereographic / 180 degrees | 52.96 px | 0.97 px | 3.25 px |
| Stereographic / 200 degrees | 65.38 px | 1.94 px | 6.82 px |
| Rectilinear / 90 degrees | 41.55 px | 0.29 px | 0.90 px |
| Rectilinear / 120 degrees | 77.77 px | 2.18 px | 7.47 px |
| Rectilinear / 160 degrees | 141.35 px | 23.69 px | 110.49 px |

These are simulations of ideal projection families, not measurements of lens
brands. Errors scale with image size. The 160-degree rectilinear one-term fit
even folds back on itself, illustrating why an invertibility check is needed.

The second experiment jointly fits orientation, optical centre, scale and two
radial terms. The synthetic lens has modest radial distortion, an off-centre
optical axis, and a roughly 2,066-pixel circle. Each trial uses 100 stars with
0.7-pixel Gaussian noise per coordinate, starts at the correct parameters, and
validates on unused, noise-free positions. Reported RMS values are medians
over 12 independent samples/noise realizations. Good initialization deliberately
isolates parameter stability from initial matching and optimization failure.

| Visible sensor / calibration coverage | Training RMS | Unused-star RMS |
| --- | ---: | ---: |
| Whole circle | 0.97 px | 0.19 px |
| 4:3, clipped circle | 0.96 px | 0.20 px |
| 16:9, clipped circle | 0.96 px | 0.22 px |
| Portrait, clipped circle | 0.96 px | 0.24 px |
| Central rectangular crop, no circle boundary visible | 0.98 px | 0.19 px |
| Off-centre rectangular crop | 0.97 px | 0.21 px |
| Full sensor, calibration stars only within central 300 px radius | 0.97 px | 204.54 px |
| Full sensor, calibration stars in a 60 px horizontal strip | 0.96 px | 0.44 px |

Crop validation uses only the visible rectangle. The central-patch and strip
cases deliberately validate the full circle, beyond the calibration region.
The central-patch result is from an unconstrained experimental fit, not a
failure of the current production solver. Five of those twelve fitted curves
also become nonmonotonic before the full-circle edge. In contrast, the thin
strip spans many radii and still constrains this particular radial model;
coverage quality cannot be reduced to a simple percentage of image area.

With synthetic asymmetric decentring, the radial model still leaves 5.18 px
RMS and 12.96 px maximum validation error despite also fitting pose and centre.
This is a counterexample to radial-only correction, not a physical simulation
of the user's dome. A rectangular sensor with square pixels needs no special
distortion model; unequal resizing of the two image axes is a different case.

## Limits in the current implementation

- `projection.py` and VirtualSky use an ideal equisolid curve. Both directions
  of the browser mapping, the Python projection, and the inverse rays used for
  pointing recovery would need to agree on any new lens model.
- `orientation.py` builds its initial star patterns using that ideal curve.
  Fitting distortion after a successful match does not solve cases where the
  original curve prevents finding the correct match in the first place.
- Recovery uses the front lens hemisphere, and VirtualSky's fisheye inverse
  rejects points beyond its 180-degree reference circle. A tilted lens can see
  above-horizon stars more than 90 degrees from its axis; supporting those
  requires changes to the domain and clipping too. A zenith-facing 200-degree
  lens may still work for its above-horizon sky without using that extra rim.
- The current catalogue is limited to magnitude 4.5 and accepted solutions need
  at least 40 matches. Across 72 sampled sky phases at latitudes -34, 0 and 53,
  a zenith-centred 30-degree circular field contained only 5-38 catalogue stars;
  a 60-degree field contained 36-103. Counts already exclude altitudes below
  10 degrees. These are geometrically available stars, before weather, sensor
  clipping or detection losses, and the samples include daylight phases.
  Thus even a perfect lens model cannot make the current solver general for
  narrow fields. Relaxing existing confidence gates is not a justified fix.
- Existing scale-dependent match tolerances and diameter/centre search bounds
  would also need validation for narrow or unusually cropped fields.
- Lens calibration cannot restore lost star detail from blur, flare or dew.
  It must not silently absorb a wrong image time or location. Multiple frames
  with stars moving across the visible sensor could improve coverage; regions
  permanently hidden by buildings still cannot be validated from those stars.
  Automated all-sky calibration using star tracks is also described in the
  [Auto-Cal preprint, 2025](https://arxiv.org/abs/2508.17146).

## Recommended scope

Keep the tilt PR unchanged. A separate, conservative extension could try one
radial term after a reliable match, adding a second only when independent
validation and coverage support it. Keep the old mapping when the additional
parameters are unconstrained or do not improve unused-star alignment. Report
coverage and edge validation, not just RMS of accepted fitting stars. Enforce
a monotonic, invertible mapping throughout the domain actually used to draw
the overlay, and persist the model and its valid domain together.

Broader lens support would need either suitable base projection selection or
a more flexible angle-to-radius model, with corresponding matching changes.
Asymmetric correction and unequal pixel scales should be separate, evidence-led
extensions. More free parameters alone do not guarantee reliable calibration.
Existing zenith installations should retain the exact legacy mapping by default.

To reproduce with the project's environment from this worktree on Windows:

```powershell
& '../indi-allsky/.venv-test/Scripts/python.exe' -m pytest tests/lens_solver/test_distortion_feasibility.py -q --basetemp=.pytest-temp-distortion
& '../indi-allsky/.venv-test/Scripts/python.exe' -m tests.lens_solver.test_distortion_feasibility
```

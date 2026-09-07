const assert = require('node:assert/strict');
const {test} = require('node:test');
const {makeSky} = require('./virtualsky_harness.cjs');
const calibration = require('../../indi_allsky/flask/static/js/virtualsky-calibration.js');

function model() {
    const coefficients = Array.from({length: 10}, () => [0, 0]);
    coefficients[3] = [0.003, 0];
    coefficients[4] = [0, 0.005];
    coefficients[6] = [0.008, 0];
    coefficients[8] = [0.008, 0];
    return {version: 1, coefficients, bounds: [-0.75, -0.7, 0.85, 0.75],
        geometry: [200, 0, 0, 1000, 0, 0, 90, 0], image_size: [1200, 900],
        context: [53, 11, 0], camera_uuid: 'camera'};
}

test('image mask follows the photo, including asymmetric borders, rather than the solved axis', () => {
    const sky = makeSky(), forward = sky.azel2xy;
    let draws = 0;
    const circles = [], clips = [];
    const draw = sky.drawImmediate = function() { draws++; return this; };
    sky.ctx = {save() {}, restore() {}, beginPath() {}, clearRect() {}, clip() { clips.push(draws); },
        arc(...args) { circles.push(args.slice(0, 3)); }};
    const mask = [2200, 36, 3, 100, 80, 70, 0, 0];
    // Actual 2406x2350 photo: mask centre 1204,1212, radius 1100.
    // The solved circle instead has centre 1193,1209 and radius 1109.
    for (const size of [2218, 554.5, 2218]) {
        for (let i = 0; i < 3; i++) calibration.maskImage(sky, mask, [2406, 2350], 1, [2218, -10, -34]);
        sky.wide = sky.tall = size;
        assert.equal(sky.drawImmediate(), sky);
        const [x, y, r] = circles.at(-1), scale = size/2218;
        assert.ok(Math.abs(x-1120*scale) < 1e-10);
        assert.ok(Math.abs(y-1112*scale) < 1e-10);
        assert.ok(Math.abs(r-1100*scale) < 1e-10);
    }
    assert.equal(draws, 3); // wrapping cannot accumulate on repeated polls
    assert.deepEqual(clips, [0, 1, 2]); // clip before drawing, preserving interior colours
    assert.equal(sky.azel2xy, forward);
    calibration.install(sky, model());
    const corrected = sky.azel2xy(1, .8, 1000, 1000);
    calibration.maskImage(sky, mask, [2406, 2350], 1, [2218, -10, -34]);
    assert.deepEqual(sky.azel2xy(1, .8, 1000, 1000), corrected);
    calibration.maskImage(sky, null, [2406, 2350], 1, [2218, -10, -34]);
    assert.equal(sky.drawImmediate, draw);
});

test('mask supports binned cropped rectangular images and disappears when metadata is unavailable', () => {
    const sky = makeSky();
    let circle;
    const draw = sky.drawImmediate = () => sky;
    sky.ctx = {save() {}, restore() {}, beginPath() {}, clearRect() {}, clip() {}, arc(...args) { circle = args; }};
    sky.wide = sky.tall = 500;
    // 600x400 crop scaled to 300x200, then borders top=10, right=20.
    // Mask radius 500/2/2*.5 = 62.5; offsets divide by binning first.
    calibration.maskImage(sky, [500, -13, 15, 50, 10, 20, 0, 0], [320, 210], 2, [500, 0, 0]);
    sky.drawImmediate();
    assert.deepEqual(circle.slice(0, 3), [237, 251.5, 62.5]);
    for (const binning of [undefined, null, 0, -1, 1.5, NaN, '2']) {
        calibration.maskImage(sky, [500, 0, 0, 50, 0, 0, 0, 0], [320, 210], binning, [500, 0, 0]);
        assert.equal(sky.drawImmediate, draw);
    }
    for (const mask of [[], [500], [500, 0, 0, 0, 0, 0, 0, 0], [NaN, 0, 0, 100, 0, 0, 0, 0],
        [500, 0, 0, 100, 0, 400, 0, 0]]) {
        calibration.maskImage(sky, mask, [320, 210], 1, [500, 0, 0]);
        assert.equal(sky.drawImmediate, draw);
    }
});

test('disabled calibration leaves the exact legacy functions in place', () => {
    const sky = makeSky(), forward = sky.azel2xy, inverse = sky.projection.xy2azel;
    calibration.install(sky, null);
    assert.equal(sky.azel2xy, forward);
    assert.equal(sky.projection.xy2azel, inverse);
});

test('reusing a sky cannot compound corrections and disabling restores both directions', () => {
    const sky = makeSky({fisheye_altitude: 87.42, fisheye_azimuth: 177.04});
    const forward = sky.azel2xy, inverse = sky.projection.xy2azel;
    calibration.install(sky, model());
    const expected = sky.azel2xy(1, 0.7, 1000, 1000);
    for (let i = 0; i < 50; i++) {
        sky.init({projection: 'fisheye'});
        calibration.install(sky, model());
        assert.deepEqual(sky.azel2xy(1, 0.7, 1000, 1000), expected);
    }
    // North is behind this slightly south-pointing lens; its label can still
    // request an unclipped position while normal stars retain their clipping.
    assert.ok(Number.isNaN(sky.azel2xy(0, 0, 1000, 1000).x));
    assert.ok(Number.isFinite(sky.azel2xy(0, 0, 1000, 1000, true).x));
    calibration.install(sky, null);
    assert.equal(sky.azel2xy, forward);
    assert.equal(sky.projection.azel2xy, forward);
    assert.equal(sky.projection.xy2azel, inverse);
});

test('stale geometry, camera, location, time and size cannot reuse calibration', () => {
    const m = model();
    const args = [m, m.geometry, m.image_size, m.context, m.camera_uuid];
    assert.equal(calibration.compatible(...args), true);
    for (const i of [1, 2, 3]) {
        const altered = [...args];
        altered[i] = [...args[i]];
        altered[i][0] += 1;
        assert.equal(calibration.compatible(...altered), false);
    }
    assert.equal(calibration.compatible(...args.slice(0, 4), 'other'), false);
    const bad = model();
    bad.geometry[0] = NaN;
    assert.equal(calibration.compatible(bad, ...args.slice(1)), false);
    bad.geometry = m.geometry;
    bad.coefficients[0] = [0.5, 0];
    assert.equal(calibration.compatible(bad, ...args.slice(1)), false);
});

for (const asset of ['virtualsky.js', 'virtualsky.min.js']) {
    test(`${asset}: forward and inverse stay consistent through tilt, scaling and taper`, () => {
        const m = model();
        for (const altitude of [0, 20, 54, 90]) for (const size of [400, 1000, 2200]) {
            const options = {fisheye_altitude: altitude, fisheye_azimuth: 123, az: 70,
                width: size, height: size};
            const sky = makeSky(options, asset), original = makeSky(options, asset);
            calibration.install(sky, m);
            for (let az = 0; az < 360; az += 30) for (const el of [15, 40, 70, 90]) {
                const a = (az-sky.az_off)*Math.PI/180, e = el*Math.PI/180;
                const p = sky.azel2xy(a, e, size, size), before = original.azel2xy(a, e, size, size);
                if (!Number.isFinite(before.x)) {
                    assert.ok(Number.isNaN(p.x));
                    continue;
                }
                const d = calibration.delta(m, (before.x-size/2)/(size/2), (before.y-size/2)/(size/2));
                assert.ok(Math.abs(p.x-before.x-d[0]*size/2) < 1e-8);
                const expected = original.xy2radec(before.x, before.y);
                const actual = sky.xy2radec(p.x, p.y);
                if (!expected) continue; // numerical points just beyond the horizon circle
                assert.ok(actual);
                assert.ok(Math.abs(actual.dec-expected.dec) < 1e-8);
                assert.ok(Math.abs(Math.cos(actual.ra)-Math.cos(expected.ra)) < 1e-8);
            }
        }
        assert.deepEqual(calibration.delta(m, 3, 3), [0, 0]);
    });
}

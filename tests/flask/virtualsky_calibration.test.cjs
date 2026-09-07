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

test('disabled calibration leaves the exact legacy functions in place', () => {
    const sky = makeSky(), forward = sky.azel2xy, inverse = sky.projection.xy2azel;
    calibration.install(sky, null);
    assert.equal(sky.azel2xy, forward);
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

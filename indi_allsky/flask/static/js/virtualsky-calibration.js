/* Optional displacement of the overlay only. Keep the polynomial and taper
 * in sync with lens_solver/calibration.py. No camera pixels are resampled. */
(function(root) {
    'use strict';
    function delta(model, u, v) {
        const b = model.bounds;
        let weight = 1;
        for (const [x, lo, hi] of [[u, b[0], b[2]], [v, b[1], b[3]]]) {
            const t = Math.min(1, Math.max(0, lo-x, x-hi) / 0.15);
            weight *= 1-t*t*(3-2*t);
        }
        const t = Math.min(1, Math.max(0, Math.hypot(u, v)-1) / 0.15);
        weight *= 1-t*t*(3-2*t);
        if (!weight) return [0, 0];
        const basis = [1, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v];
        const d = [0, 0];
        basis.forEach((value, i) => {
            d[0] += value*model.coefficients[i][0]*weight;
            d[1] += value*model.coefficients[i][1]*weight;
        });
        return d;
    }

    function compatible(model, geometry, size, context, uuid) {
        if (!model || model.version !== 1 || model.camera_uuid !== uuid) return false;
        for (const [saved, current] of [[model.geometry, geometry], [model.image_size, size],
                                        [model.context, context]]) {
            if (!Array.isArray(saved) || saved.length !== current.length ||
                saved.some((n, i) => !Number.isFinite(n) || !Number.isFinite(current[i])
                    || Math.abs(n-current[i]) > 1e-8)) return false;
        }
        return Array.isArray(model.bounds) && model.bounds.length === 4
            && model.bounds.every(n => Number.isFinite(n) && Math.abs(n) <= 3)
            && model.bounds[2]-model.bounds[0] >= 0.15 && model.bounds[3]-model.bounds[1] >= 0.15
            && Array.isArray(model.coefficients) && model.coefficients.length === 10
            && model.coefficients.every(row => Array.isArray(row) && row.length === 2
                && row.every(n => Number.isFinite(n) && Math.abs(n) <= 1)) && safe(model);
    }

    function safe(model) {
        // Also check imported camera metadata, which may bypass our Save endpoint.
        const b = model.bounds, step = 1e-5;
        for (let i = 0; i < 49; i++) for (let j = 0; j < 49; j++) {
            const u = b[0]-0.15+(b[2]-b[0]+0.3)*i/48;
            const v = b[1]-0.15+(b[3]-b[1]+0.3)*j/48;
            if (Math.hypot(...delta(model, u, v)) > 0.06) return false;
            const left = delta(model, u-step, v), right = delta(model, u+step, v);
            const up = delta(model, u, v-step), down = delta(model, u, v+step);
            if (Math.hypot(right[0]-left[0], right[1]-left[1], down[0]-up[0], down[1]-up[1])
                / (2*step) >= 0.45) return false;
        }
        return true;
    }

    function install(sky, model) {
        // The page reuses its sky instance. Always restore the base projection
        // before toggling or replacing a model so corrections cannot accumulate.
        if (sky.calibrationProjection) {
            sky.azel2xy = sky.projection.azel2xy = sky.calibrationProjection.forward;
            sky.projection.xy2azel = sky.calibrationProjection.inverse;
        }
        if (!model) return;
        const forward = sky.azel2xy;
        const inverse = sky.projection.xy2azel;
        sky.calibrationProjection = {forward, inverse};
        sky.azel2xy = sky.projection.azel2xy = function(az, el, w, h, unclipped) {
            const p = forward.call(this, az, el, w, h, unclipped);
            if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return p;
            const r = h/2;
            const d = delta(model, (p.x-w/2)/r, (p.y-h/2)/r);
            p.x += d[0]*r;
            p.y += d[1]*r;
            return p;
        };
        sky.projection.xy2azel = function(x, y, w, h) {
            const r = h/2, target = [(x-w/2)/r, (y-h/2)/r];
            if (!target.every(Number.isFinite)) return undefined;
            let u = target[0], v = target[1];
            // Server validation bounds the displacement gradient below 0.45,
            // so this fixed-point inverse converges even across the taper.
            for (let i = 0; i < 30; i++) {
                const d = delta(model, u, v);
                const nu = target[0]-d[0], nv = target[1]-d[1];
                if (Math.hypot(nu-u, nv-v) < 1e-10) {
                    return inverse.call(this, w/2+nu*r, h/2+nv*r, w, h);
                }
                u = nu; v = nv;
            }
            return undefined;
        };
    }
    const api = {delta, compatible, install};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.VirtualSkyCalibration = api;
})(typeof window !== 'undefined' ? window : globalThis);

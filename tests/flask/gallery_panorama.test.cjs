const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('panorama link follows the current slide and clears missing panoramas', () => {
    const template = fs.readFileSync(path.join(__dirname,
        '../../indi_allsky/flask/templates/gallery.html'), 'utf8');
    const registration = template.match(/lightbox\.pswp\.ui\.registerElement\(\{\s*name: 'panorama-button',[\s\S]*?\n        \}\);/)[0];
    let control, change, panoramaId;
    vm.runInNewContext(registration.replace(/{{[^}]+}}/g, '/view_panorama'), {
        lightbox: { pswp: { ui: { registerElement(options) { control = options; } } } },
    });
    const link = {
        style: {},
        setAttribute(key, value) { this[key] = value; },
        removeAttribute(key) { delete this[key]; },
    };
    control.onInit(link, {
        on(event, callback) { assert.equal(event, 'change'); change = callback; },
        currSlide: { data: { element: {
            getAttribute(key) { assert.equal(key, 'data-panorama_id'); return panoramaId; },
        } } },
    });
    assert.equal(link.target, '_blank');
    assert.equal(link.rel, 'noopener');
    for (panoramaId of ['21', '', '23']) {
        change();
        assert.equal(link.style.display, panoramaId ? '' : 'none');
        assert.equal(link.href, panoramaId ? `/view_panorama?id=${panoramaId}` : undefined);
    }
});

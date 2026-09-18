// Run through pytest by test_allsky_map_config.py.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const template = fs.readFileSync(path.join(__dirname, '../../indi_allsky/flask/templates/config.html'), 'utf8');
const start = template.indexOf('const field_names = [');
const end = template.indexOf('function setBrowserLocationStatus(');
assert.ok(start >= 0 && end > start);
const script = template.slice(start, end);

for (const [latitude, longitude] of [['50.12678', '8.07654'], ['0', '0'], ['', '']]) {
    test(`save submits map overrides ${JSON.stringify([latitude, longitude])}`, () => {
        const values = {
            '#ALLSKYMAP__MAP_LATITUDE': latitude,
            '#ALLSKYMAP__MAP_LONGITUDE': longitude,
            '#LOCATION_LATITUDE': '50.1234567',
            '#LOCATION_LONGITUDE': '8.1234567',
        };
        const handlers = new Map();
        const requests = [];
        const $ = (selector) => ({
            val: () => values[selector] ?? '',
            prop: () => false,
            on: (event, handler) => handlers.set(`${selector}:${event}`, handler),
            css() {}, hide() {}, remove() {}, removeClass() {}, attr() {}, html() {},
        });
        $.ajax = (request) => requests.push(request);
        vm.runInNewContext(script, { $, successMessage: $('success') });
        handlers.get('#form_config:submit')();
        assert.equal(requests.length, 1);
        const payload = JSON.parse(requests[0].data);
        assert.equal(payload.ALLSKYMAP__MAP_LATITUDE, latitude);
        assert.equal(payload.ALLSKYMAP__MAP_LONGITUDE, longitude);
        assert.equal(payload.LOCATION_LATITUDE, '50.1234567');
        assert.equal(payload.LOCATION_LONGITUDE, '8.1234567');
    });
}

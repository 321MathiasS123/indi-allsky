const test = require('node:test');
const assert = require('node:assert/strict');
const {formatStatus, mount} = require('../../indi_allsky/flask/static/js/syncapi-on-demand.js');

function harness() {
    function element() {
        return {children: [], handlers: {}, appendChild(child) { this.children.push(child); },
            addEventListener(event, handler) { this.handlers[event] = handler; },
            querySelectorAll() { return this.children.flatMap(label => label.children || []).filter(node => node.type === 'checkbox' && node.checked); }};
    }
    const nodes = Object.fromEntries(['start', 'cancel', 'types', 'status', 'error'].map(key => [key, element()]));
    const document = {getElementById: id => nodes[id.replace('syncapi-run-', '')],
        createElement: element, createTextNode: text => ({textContent: text})};
    const panel = {dataset: {url: '/indi-allsky/ajax/syncapi/run', csrf: 'token'}};
    return {nodes, document, panel};
}

test('progress describes saved results, cutoff and pending cancellation', () => {
    const text = formatStatus({enabled: true, message: 'Interrupted', completed: 2, total: 4,
        skipped: 1, files: 3, bytes: 1048576, cutoff: '2026-09-13T12:00:00', cancel_requested: true});
    assert.match(text, /2 of 4/);
    assert.match(text, /1.0 MiB/);
    assert.match(text, /2026-09-13 12:00:00/);
    assert.match(text, /Cancellation requested/);
});

test('polling only reads local status; start and cancel are explicit CSRF-protected requests', async () => {
    const {nodes, document, panel} = harness();
    const requests = [], scheduled = [];
    const fetcher = async (url, options) => {
        requests.push({url, options});
        const payload = options.body && JSON.parse(options.body);
        return {ok: true, json: async () => ({enabled: true, active: Boolean(payload), task_id: 42,
            state: payload ? 'queued' : 'idle', types: [{id: 'image', label: 'Images', selected: true}, {id: 'rawimage', label: 'RAW', selected: false}]})};
    };
    await mount(panel, document, fetcher, (fn, delay) => scheduled.push({fn, delay}));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].options.method, undefined);
    assert.equal(nodes.start.disabled, false);
    await nodes.start.handlers.click();
    assert.deepEqual(JSON.parse(requests[1].options.body), {action: 'start', types: ['image']});
    assert.equal(requests[1].options.headers['X-CSRFToken'], 'token');
    assert.equal(nodes.start.disabled, true);
    await nodes.cancel.handlers.click();
    assert.deepEqual(JSON.parse(requests[2].options.body), {action: 'cancel', task_id: 42});
    assert.ok(requests.every(item => item.url === panel.dataset.url));
    assert.equal(scheduled[0].delay, 5000);
});

test('failed start remains visible and does not trigger automatic retry', async () => {
    const {nodes, document, panel} = harness();
    let requests = 0;
    const fetcher = async () => {
        requests++;
        return {ok: requests === 1, json: async () => requests === 1 ?
            {enabled: true, active: false, types: [{id: 'image', label: 'Images', selected: true}]} : {error: 'Apply configuration first'}};
    };
    await mount(panel, document, fetcher, () => {});
    await nodes.start.handlers.click();
    assert.equal(nodes.error.textContent, 'Apply configuration first');
    assert.equal(requests, 2);
    assert.equal(nodes.start.disabled, false);
});

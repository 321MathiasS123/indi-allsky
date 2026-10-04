const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../indi_allsky/flask/static/js/custom_keogram.js'), 'utf8');

async function setup(responses, restore = false, search = '') {
    // Execute the shipped script with controlled network replies and manual timers.
    if (!restore) responses.unshift({data: {state: 'EMPTY', preview: null}});
    const elements = {};
    for (const id of ['custom-keogram-form', 'custom-keogram-fields', 'keogram-start', 'keogram-end',
        'custom-keogram-status', 'custom-keogram-result', 'custom-keogram-image', 'custom-keogram-open', 'custom-keogram-download',
        'custom-keogram-range']) {
        elements[id] = {value: '', disabled: false, hidden: true, handlers: {}, validity: '',
            classList: {toggle() {}},
            addEventListener(name, fn) { this.handlers[name] = fn; },
            setCustomValidity(text) { this.validity = text; }};
    }
    elements['keogram-start'].value = '2026-10-01T15:00';
    elements['keogram-end'].value = '2026-10-01T22:00';
    elements['custom-keogram-form'].reportValidity = () => !elements['keogram-end'].validity;
    const requests = [], timers = [];
    const window = {location: {href: 'http://localhost/custom_keogram' + search},
        history: {replaceState(a, b, url) { window.location.href = String(url); }}};
    const context = {document: {getElementById(id) { return elements[id]; }}, window, URL, AbortSignal,
        setTimeout(fn) { timers.push(fn); },
        async fetch(url, options) {
            requests.push({url: String(url), options});
            const response = responses.shift();
            if (response instanceof Error) throw response;
            assert.ok(response, 'Unexpected network request');
            return {ok: !response.status || response.status < 400, status: response.status || 200,
                redirected: response.redirected || false, json: async () => {
                    if (response.jsonError) throw new SyntaxError('Unexpected HTML response');
                    return response.data;
                }};
        }};
    vm.createContext(context);
    vm.runInContext(source, context);
    context.initCustomKeogram({endpoint: '/ajax/custom_keogram', cameraId: 1, csrfToken: 'csrf', enabled: true});
    await tick();
    return {elements, requests, timers, window,
        submit: () => elements['custom-keogram-form'].handlers.submit({preventDefault() {}})};
}

const running = {task_id: 17, state: 'RUNNING', frames: 25, skipped: 2, total: 100,
    start: '2026-10-01T15:00', end: '2026-10-01T22:00'};
const saved = {...running, state: 'SUCCESS', message: 'Created from 98 images.', frames: 98,
    first: '2026-10-01 15:00:00', last: '2026-10-01 22:00:00', image_url: '/ajax/custom_keogram?task_id=17&image=1'};
// Let fetch/JSON promises settle without advancing the polling timer.
const tick = () => new Promise(resolve => setImmediate(resolve));

test('rejects reversed dates without making a request', async () => {
    const h = await setup([]);
    h.elements['keogram-end'].value = '2026-10-01T14:00';
    await h.submit();
    assert.match(h.elements['keogram-end'].validity, /later/);
    assert.equal(h.requests.length, 1); // Only the initial restoration request.
});

test('submits exact camera-local dates, prevents double submit and polls to download', async () => {
    const h = await setup([{data: {task_id: 17}}, {data: running}, {data: {...saved, preview: saved}}]);
    await h.submit();
    assert.deepEqual(JSON.parse(h.requests[1].options.body), {
        camera_id: 1, start: '2026-10-01T15:00', end: '2026-10-01T22:00'});
    assert.equal(h.requests[1].options.headers['X-CSRFToken'], 'csrf');
    assert.equal(h.elements['custom-keogram-fields'].disabled, true);
    await h.submit();
    assert.equal(h.requests.length, 3);
    h.timers.shift()();
    await tick();
    assert.equal(h.elements['custom-keogram-fields'].disabled, false);
    h.elements['custom-keogram-image'].onload();
    assert.equal(h.elements['custom-keogram-result'].hidden, false);
    assert.match(h.elements['custom-keogram-download'].href, /&download=1$/);
    assert.equal(h.timers.length, 0);
});

test('resumes after reload and retries a transient status failure', async () => {
    const h = await setup([new Error('offline'), {data: {...running, state: 'FAILED', message: 'No readable images.'}}], true);
    await tick();
    assert.match(h.elements['custom-keogram-status'].textContent, /Retrying/);
    assert.equal(h.elements['custom-keogram-fields'].disabled, true);
    h.timers.shift()();
    await tick();
    assert.equal(h.elements['custom-keogram-status'].textContent, 'No readable images.');
    assert.equal(h.elements['custom-keogram-fields'].disabled, false);
    assert.equal(h.timers.length, 0);
});

test('restores the selected camera independently of stale task URL parameters', async () => {
    const h = await setup([], false, '?task_id=17&camera_id=2');
    const url = new URL(h.requests[0].url);
    assert.equal(url.searchParams.get('camera_id'), '1');
    assert.equal(url.searchParams.has('task_id'), false);
});

test('expired previews and login redirects terminate polling with useful errors', async () => {
    for (const response of [{status: 410, data: {message: 'This temporary preview has expired.'}},
        {redirected: true}]) {
        const h = await setup([response], true);
        await tick();
        assert.match(h.elements['custom-keogram-status'].textContent, /expired/);
        assert.equal(h.elements['custom-keogram-fields'].disabled, false);
        assert.equal(h.timers.length, 0);
    }
});

test('queue errors restore the form and malformed polling responses retry', async () => {
    const h = await setup([{status: 400, data: {message: 'No saved images were found.'}}]);
    await h.submit();
    assert.equal(h.elements['custom-keogram-fields'].disabled, false);
    assert.equal(h.elements['custom-keogram-status'].textContent, 'No saved images were found.');
    const malformed = await setup([{data: {unexpected: true}}], true);
    await tick();
    assert.equal(malformed.timers.length, 1);
    assert.match(malformed.elements['custom-keogram-status'].textContent, /Retrying/);
});

test('HTTP errors keep their retry policy even with HTML or empty responses', async () => {
    for (const response of [{status: 403, jsonError: true}, {status: 404, data: null},
        {status: 429, jsonError: true}, {status: 503, jsonError: true}, {jsonError: true}]) {
        const h = await setup([response], true);
        await tick();
        const retries = ![403, 404].includes(response.status);
        assert.equal(h.timers.length, retries ? 1 : 0);
        assert.equal(h.elements['custom-keogram-fields'].disabled, retries);
        assert.match(h.elements['custom-keogram-status'].textContent, retries ? /Retrying/ : /request failed/);
    }
});

test('malformed submission replies restore the form without submitting another job', async () => {
    for (const response of [{status: 400, jsonError: true}, {data: null}]) {
        const h = await setup([response]);
        await h.submit();
        assert.equal(h.elements['custom-keogram-fields'].disabled, false);
        assert.match(h.elements['custom-keogram-status'].textContent, /Please try again/);
        assert.equal(h.requests.length, 2);
        assert.equal(h.timers.length, 0);
    }
});

test('restores the last image and keeps its range visible during a failed replacement', async () => {
    const h = await setup([{data: {...saved, preview: saved}}, {data: {task_id: 18}},
        {data: {...running, task_id: 18, start: '2026-10-01T21:00', preview: saved}},
        {data: {...running, task_id: 18, state: 'FAILED', message: 'Generation failed.', preview: saved}}], true);
    const image = h.elements['custom-keogram-image'];
    image.onload();
    const caption = h.elements['custom-keogram-range'].textContent;
    assert.match(caption, /2026-10-01 15:00 to 2026-10-01 22:00/);
    h.elements['keogram-start'].value = '2026-10-01T21:00';
    await h.submit();
    assert.equal(h.elements['custom-keogram-result'].hidden, false);
    assert.equal(h.elements['custom-keogram-range'].textContent, caption);
    assert.equal(h.timers.length, 1);
    h.timers.shift()();
    await tick();
    assert.equal(h.elements['custom-keogram-result'].hidden, false);
    assert.equal(h.elements['custom-keogram-range'].textContent, caption);
    assert.equal(h.elements['custom-keogram-status'].textContent, 'Generation failed.');
    assert.equal(h.elements['custom-keogram-fields'].disabled, false);
});

test('a fresh visit recovers an active job and replaces the preview when it succeeds', async () => {
    const newer = {...saved, task_id: 18, start: '2026-10-01T21:00', first: '2026-10-01 21:00:00',
        image_url: '/ajax/custom_keogram?task_id=18&image=1'};
    const h = await setup([{data: {...running, task_id: 18, preview: saved}},
        {data: {...newer, preview: newer}}], true);
    const image = h.elements['custom-keogram-image'];
    image.onload();
    assert.equal(h.elements['custom-keogram-fields'].disabled, true);
    h.timers.shift()();
    await tick();
    image.onload();
    assert.match(h.requests[1].url, /task_id=18/);
    assert.match(h.elements['custom-keogram-download'].href, /task_id=18/);
    assert.match(h.elements['custom-keogram-range'].textContent, /2026-10-01 21:00 to/);
    assert.equal(h.elements['custom-keogram-fields'].disabled, false);
    assert.equal(h.requests.some(request => request.options.method === 'POST'), false);
});

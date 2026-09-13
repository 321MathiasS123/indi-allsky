const test = require('node:test');
const assert = require('node:assert/strict');
const {formatStatus, mount, schedulePayload} = require('../../indi_allsky/flask/static/js/syncapi-on-demand.js');

function harness() {
    function element() {
        return {children: [], handlers: {}, appendChild(child) { this.children.push(child); },
            addEventListener(event, handler) { this.handlers[event] = handler; },
            querySelectorAll(selector) { return this.children.flatMap(label => label.children || []).filter(node => node.type === 'checkbox' && (selector === 'input' || node.checked)); }};
    }
    const nodes = Object.fromEntries(['start', 'cancel', 'types', 'status', 'error', 'schedule-controls',
        'schedule-enabled', 'schedule-interval', 'schedule-delay'].map(key => [key, element()]));
    nodes.types.children = [{children: [{type: 'checkbox', value: 'image', checked: true}]},
        {children: [{type: 'checkbox', value: 'rawimage', checked: false}]}];
    nodes['schedule-enabled'].checked = false;
    nodes['schedule-interval'].value = '10';
    nodes['schedule-delay'].value = '3';
    const document = {getElementById: id => nodes[id.replace('syncapi-run-', '')],
        createElement: element, createTextNode: text => ({textContent: text})};
    const panel = {dataset: {url: '/indi-allsky/ajax/syncapi/run', csrf: 'token'}};
    return {nodes, document, panel};
}

test('progress describes saved results, cutoff and pending cancellation', () => {
    const text = formatStatus({enabled: true, message: 'Interrupted', completed: 2, total: 4,
        skipped: 1, files: 3, bytes: 1048576, cutoff: '2026-09-13T12:00:00.157618', cancel_requested: true});
    assert.match(text, /2 of 4/);
    assert.match(text, /1.0 MiB/);
    assert.match(text, /2026-09-13 12:00:00/);
    assert.doesNotMatch(text, /157618/);
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
    const requests = [], scheduled = [];
    const fetcher = async (url, options) => {
        requests.push(options);
        return {ok: !options.body, json: async () => !options.body ?
            {enabled: true, active: false, types: [{id: 'image', label: 'Images', selected: true}]} : {error: 'Apply configuration first'}};
    };
    await mount(panel, document, fetcher, fn => scheduled.push(fn));
    await nodes.start.handlers.click();
    assert.equal(nodes.error.textContent, 'Apply configuration first');
    assert.equal(requests.length, 2);
    await scheduled.shift()();
    assert.equal(requests.length, 3);
    assert.equal(requests[2].method, undefined);
    assert.equal(nodes.error.textContent, 'Apply configuration first');
    assert.equal(nodes.start.disabled, false);
});

test('configuration payload includes current switch, timings and checkboxes without uploading', () => {
    const {nodes, document} = harness();
    nodes['schedule-enabled'].checked = true;
    nodes['schedule-interval'].value = '5';
    nodes['schedule-delay'].value = '0';
    nodes.types.querySelectorAll('input')[1].checked = true;
    assert.deepEqual(schedulePayload(document), {enabled: true, interval: 5, delay: 0, types: ['image', 'rawimage']});
    nodes['schedule-delay'].value = '';
    assert.equal(schedulePayload(document).delay, null, 'An empty delay must not silently become zero');
});

test('Sync now uses unsaved checkboxes without changing scheduled content, even across newer polls', async () => {
    const {nodes, document, panel} = harness();
    const requests = [], polls = [];
    let revision = 'original';
    const fetcher = async (url, options) => {
        requests.push(options);
        return {ok: true, json: async () => ({enabled: true, active: false,
            schedule: {settings: {enabled: false, interval: 10, delay: 3, types: ['image'], revision},
                message: 'Waiting for receiver.', next_action: '2026-09-13T20:15:00+02:00'}})};
    };
    // Edits made before the first poll returns must also survive.
    nodes.types.querySelectorAll('input')[0].checked = false;
    nodes.types.querySelectorAll('input')[1].checked = true;
    nodes['schedule-interval'].value = '7';
    await mount(panel, document, fetcher, fn => polls.push(fn));
    revision = 'changed elsewhere';
    await polls.shift()();
    assert.equal(nodes['schedule-interval'].value, '7');
    await nodes.start.handlers.click();
    assert.deepEqual(JSON.parse(requests[2].body), {action: 'start', types: ['rawimage']});
    assert.equal(requests.filter(options => options.method === 'POST').length, 1);
    assert.match(nodes.status.textContent, /Waiting for receiver/);
    assert.match(nodes.status.textContent, /Next check: 2026-09-13 20:15:00/);
    nodes.types.querySelectorAll('input')[1].checked = false;
    await nodes.start.handlers.click();
    assert.equal(requests.length, 3);
    assert.match(nodes.error.textContent, /Select at least one/);
});

function deferred() {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return {promise, resolve};
}

for (const active of [false, true]) {
    test(`slow polling preserves controls and edits while ${active ? 'running' : 'idle'}`, async () => {
        const {nodes, document, panel} = harness();
        const polls = [], pending = deferred();
        const state = {enabled: true, active, task_id: 42, cancel_requested: false,
            types: [{id: 'image', label: 'Images', selected: true}],
            schedule: {settings: {enabled: true, interval: 10, delay: 3, types: ['image'], revision: 'saved'}}};
        const response = {ok: true, json: async () => state};
        let reads = 0;
        await mount(panel, document, async () => ++reads === 1 ? response : pending.promise, fn => polls.push(fn));
        const controls = ['start', 'cancel', 'types', 'schedule-controls'];
        const disabled = controls.map(key => nodes[key].disabled);
        nodes['schedule-interval'].value = '7';
        nodes.types.querySelectorAll('input')[0].checked = false;
        const poll = polls.shift()();
        assert.deepEqual(controls.map(key => nodes[key].disabled), disabled);
        assert.equal(nodes[active ? 'cancel' : 'start'].disabled, false);
        pending.resolve(response);
        await poll;
        assert.deepEqual(controls.map(key => nodes[key].disabled), disabled);
        assert.equal(nodes['schedule-interval'].value, '7');
        assert.equal(nodes.types.querySelectorAll('input')[0].checked, false);
    });
}

for (const staleFails of [false, true]) {
    for (const pollFinishesFirst of [false, true]) {
        test(`cancel overtakes a poll (${staleFails ? 'failed' : 'successful'}, finishes ${pollFinishesFirst ? 'before' : 'after'} command)`, async () => {
            const {nodes, document, panel} = harness();
            const polls = [], requests = [], pendingPoll = deferred(), pendingCommand = deferred();
            const running = {enabled: true, active: true, task_id: 42,
                types: [{id: 'image', label: 'Images', selected: true}],
                schedule: {settings: {enabled: true, interval: 10, delay: 3, types: ['image'], revision: 'old'}}};
            const response = value => ({ok: true, json: async () => value});
            const fetcher = async (url, options) => {
                requests.push(options);
                if (options.body) return pendingCommand.promise;
                return requests.length === 1 ? response(running) : pendingPoll.promise;
            };
            await mount(panel, document, fetcher, fn => polls.push(fn));
            const poll = polls.shift()();
            const command = nodes.cancel.handlers.click();
            assert.equal(requests.length, 3, 'Cancel is sent without waiting for the poll');
            assert.deepEqual(JSON.parse(requests[2].body), {action: 'cancel', task_id: 42});
            assert.equal(nodes.cancel.disabled, true);
            await nodes.cancel.handlers.click();
            assert.equal(requests.length, 3, 'A pending command cannot be submitted twice');
            const settlePoll = async () => {
                pendingPoll.resolve(staleFails ? {ok: false, json: async () => ({error: 'Old poll failed'})} : response(running));
                await poll;
            };
            if (pollFinishesFirst) {
                await settlePoll();
                assert.equal(nodes.cancel.disabled, true, 'A stale poll cannot release the command lock');
                await polls.shift()();
                assert.equal(requests.length, 3, 'Polling waits while a command is pending');
            }
            pendingCommand.resolve(response({...running, cancel_requested: true,
                schedule: {settings: {...running.schedule.settings, enabled: false, revision: 'cancelled'}}}));
            await command;
            if (!pollFinishesFirst) await settlePoll();
            assert.equal(nodes.cancel.disabled, true, 'A stale poll cannot undo cancellation');
            assert.equal(nodes['schedule-enabled'].checked, false);
            assert.equal(nodes.error.textContent, '');
            assert.match(nodes.status.textContent, /Cancellation requested/);
        });
    }
}

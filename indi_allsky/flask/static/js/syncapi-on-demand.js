(function (root) {
    'use strict';

    function formatStatus(state) {
        const scheduleParts = [];
        if (state.schedule) {
            if (state.schedule.message) scheduleParts.push(state.schedule.message);
            if (state.schedule.next_action) scheduleParts.push('Next check: ' + state.schedule.next_action.replace('T', ' ') + '.');
        }
        const scheduled = !state.active && state.enabled && state.schedule && state.schedule.settings && state.schedule.settings.enabled;
        let message = state.message || state.state || 'Idle';
        if (scheduled && ['cancelled', 'interrupted', 'failed', 'complete'].includes(state.state)) {
            // The saved run is history; the schedule now determines how to
            // continue. Keep its result without obsolete manual-start advice.
            message = 'Previous run: ' + message.replace(/ Press Sync now[^.]*\./g, '')
                .replace(' The schedule will check the receiver again.', '');
        }
        const parts = scheduled ? scheduleParts.concat(message) : [message];
        if (typeof state.completed === 'number') {
            parts.push(`${state.completed} of ${state.total} items completed; ${state.skipped} skipped; ${state.files} files, ${(state.bytes / 1048576).toFixed(1)} MiB sent.`);
        }
        if (state.cutoff) parts.push(`Includes completed files through ${state.cutoff.replace('T', ' ').replace(/\.\d+/, '')}.`);
        if (state.cancel_requested) parts.push('Cancellation requested; waiting for the current transfer to finish or time out.');
        if (!state.enabled) parts.push('Enable Sync API, select On demand, then save and apply.');
        if (!scheduled) parts.push(...scheduleParts);
        return parts.join('\n');
    }

    function selectedTypes(document) {
        return Array.from(document.getElementById('syncapi-run-types').querySelectorAll('input:checked'), input => input.value);
    }

    function schedulePayload(document) {
        const minutes = id => {
            const value = document.getElementById(id).value.trim();
            return value ? Number(value) : null;
        };
        return {enabled: document.getElementById('syncapi-run-schedule-enabled').checked,
            interval: minutes('syncapi-run-schedule-interval'), delay: minutes('syncapi-run-schedule-delay'),
            types: selectedTypes(document)};
    }

    function mount(panel, document, fetcher, schedule) {
        const start = document.getElementById('syncapi-run-start');
        const cancel = document.getElementById('syncapi-run-cancel');
        const choices = document.getElementById('syncapi-run-types');
        const output = document.getElementById('syncapi-run-status');
        const error = document.getElementById('syncapi-run-error');
        const controls = document.getElementById('syncapi-run-schedule-controls');
        const enabled = document.getElementById('syncapi-run-schedule-enabled');
        let state = {};
        let commandPending = false;
        let requestRevision = 0;

        function render(value) {
            state = value;
            // The server renders saved form values. Polling only updates
            // progress, never unsaved timing edits or one-off media choices.
            // Scheduler and task status are read separately; either can report
            // a run first while the scheduler hands it to the transfer worker.
            start.disabled = commandPending || !value.enabled || value.active ||
                Boolean(value.schedule && value.schedule.state === 'running');
            cancel.disabled = commandPending || !value.active || value.cancel_requested;
            choices.disabled = commandPending || value.active;
            controls.disabled = commandPending || value.active;
            output.textContent = formatStatus(value);
        }

        async function request(payload) {
            const options = {credentials: 'same-origin', cache: 'no-store'};
            if (payload) {
                options.method = 'POST';
                options.headers = {'Content-Type': 'application/json', 'X-CSRFToken': panel.dataset.csrf};
                options.body = JSON.stringify(payload);
            }
            const response = await fetcher(panel.dataset.url, options);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Unable to read synchronization status.');
            return data;
        }

        async function refresh(payload) {
            // Polls leave the controls usable. A command can overtake an older
            // poll; its revision prevents that poll from restoring stale state
            // or errors. Only commands lock controls and clear action errors.
            if (commandPending) return;
            const revision = ++requestRevision;
            if (payload) {
                commandPending = true;
                render(state);
                error.textContent = '';
            }
            try {
                const value = await request(payload);
                if (revision === requestRevision) {
                    state = value;
                    if (payload && payload.action === 'cancel' && value.schedule) {
                        enabled.checked = value.schedule.settings.enabled;
                    }
                }
            } catch (exception) {
                if (revision === requestRevision) error.textContent = exception.message;
            } finally {
                if (revision === requestRevision) {
                    commandPending = false;
                    render(state);
                }
            }
        }

        start.addEventListener('click', function () {
            const types = selectedTypes(document);
            if (!types.length) {
                error.textContent = 'Select at least one media type.';
                return;
            }
            return refresh({action: 'start', types: types});
        });
        cancel.addEventListener('click', function () {
            return refresh({action: 'cancel', task_id: state.task_id});
        });
        // A successful configuration save should not wait for the next poll.
        document.addEventListener('indi-allsky:config-saved', function () { return refresh(); });

        async function poll() {
            // This endpoint reads the Pi's saved status; polling never contacts
            // the NAS. Only explicit button handlers send commands.
            await refresh();
            schedule(poll, 5000);
        }
        return poll();
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = {formatStatus, mount, schedulePayload};
    if (root.document) {
        root.indiAllskySync = {schedulePayload};
        const panel = root.document.getElementById('syncapi-run-panel');
        if (panel) mount(panel, root.document, root.fetch.bind(root), root.setTimeout.bind(root));
    }
})(typeof window === 'undefined' ? globalThis : window);

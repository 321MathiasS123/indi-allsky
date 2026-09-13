(function (root) {
    'use strict';

    function formatStatus(state) {
        const parts = [state.message || state.state || 'Idle'];
        if (typeof state.completed === 'number') {
            parts.push(`${state.completed} of ${state.total} items completed; ${state.skipped} skipped; ${state.files} files, ${(state.bytes / 1048576).toFixed(1)} MiB sent.`);
        }
        if (state.cutoff) parts.push(`Includes completed files through ${state.cutoff.replace('T', ' ').replace(/\.\d+/, '')}.`);
        if (state.cancel_requested) parts.push('Cancellation requested; waiting for the current transfer to finish or time out.');
        if (!state.enabled) parts.push('Enable Sync API, select On demand, then save and apply.');
        return parts.join(' ');
    }

    function mount(panel, document, fetcher, schedule) {
        const start = document.getElementById('syncapi-run-start');
        const cancel = document.getElementById('syncapi-run-cancel');
        const choices = document.getElementById('syncapi-run-types');
        const output = document.getElementById('syncapi-run-status');
        const error = document.getElementById('syncapi-run-error');
        const controls = document.getElementById('syncapi-run-schedule-controls');
        const enabled = document.getElementById('syncapi-run-schedule-enabled');
        const interval = document.getElementById('syncapi-run-schedule-interval');
        const delay = document.getElementById('syncapi-run-schedule-delay');
        const save = document.getElementById('syncapi-run-schedule-save');
        const saveFeedback = document.getElementById('syncapi-run-schedule-feedback');
        const scheduleOutput = document.getElementById('syncapi-run-schedule-status');
        let state = {};
        let commandPending = false;
        let requestRevision = 0;
        let initialized = false;
        let settingsRevision;

        function render(value) {
            state = value;
            // Initialize once so status polls preserve the user's selections.
            if (!initialized && Array.isArray(value.types)) {
                value.types.forEach(function (type) {
                    const label = document.createElement('label');
                    const input = document.createElement('input');
                    label.className = 'tw:flex tw:items-center tw:gap-3 tw:p-3 tw:bg-base-100 tw:border tw:border-base-300 tw:rounded-[var(--radius-field)] tw:text-xs tw:cursor-pointer';
                    input.type = 'checkbox';
                    input.className = 'tw:checkbox tw:checkbox-primary tw:checkbox-sm tw:shrink-0';
                    input.value = type.id;
                    input.checked = type.selected;
                    label.appendChild(input);
                    label.appendChild(document.createTextNode(' ' + type.label));
                    choices.appendChild(label);
                });
                initialized = true;
            }
            if (value.schedule) {
                const settings = value.schedule.settings;
                // Preserve edits during polls; a saved revision also reflects
                // cancellation or schedule changes made in another browser.
                if (settings.revision !== settingsRevision) {
                    saveFeedback.textContent = '';
                    enabled.checked = settings.enabled;
                    interval.value = String(settings.interval);
                    delay.value = String(settings.delay);
                    choices.querySelectorAll('input').forEach(input => { input.checked = settings.types.includes(input.value); });
                    settingsRevision = settings.revision;
                }
                scheduleOutput.textContent = value.schedule.message || '';
                if (value.schedule.next_action) {
                    scheduleOutput.textContent += ' Next action: ' + value.schedule.next_action.replace('T', ' ') + '.';
                }
            }
            start.disabled = commandPending || !value.enabled || value.active;
            cancel.disabled = commandPending || !value.active || value.cancel_requested;
            choices.disabled = commandPending || value.active;
            controls.disabled = commandPending || !value.enabled || value.active;
            save.disabled = controls.disabled || !value.schedule;
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
            let scheduleSaved = false;
            if (payload) {
                commandPending = true;
                render(state);
                error.textContent = '';
                saveFeedback.textContent = payload.action === 'schedule' ? 'Saving schedule…' : '';
            }
            try {
                const value = await request(payload);
                if (revision === requestRevision) {
                    state = value;
                    scheduleSaved = payload && payload.action === 'schedule';
                }
            } catch (exception) {
                if (revision === requestRevision) error.textContent = exception.message;
            } finally {
                if (revision === requestRevision) {
                    commandPending = false;
                    render(state);
                    if (payload && payload.action === 'schedule') {
                        saveFeedback.textContent = scheduleSaved ? 'Schedule saved. Automatic synchronization is ' +
                            (state.schedule.settings.enabled ? 'enabled.' : 'disabled.') : '';
                    }
                }
            }
        }

        start.addEventListener('click', function () {
            const types = Array.from(choices.querySelectorAll('input:checked'), input => input.value);
            if (!types.length) {
                error.textContent = 'Select at least one media type.';
                return;
            }
            return refresh({action: 'start', types: types});
        });
        cancel.addEventListener('click', function () {
            return refresh({action: 'cancel', task_id: state.task_id});
        });
        save.addEventListener('click', function () {
            saveFeedback.textContent = '';
            const types = Array.from(choices.querySelectorAll('input:checked'), input => input.value);
            if (!types.length || !interval.value.trim() || !delay.value.trim()) {
                error.textContent = 'Select content types and enter both timing values.';
                return;
            }
            return refresh({action: 'schedule', enabled: enabled.checked,
                interval: Number(interval.value), delay: Number(delay.value), types: types});
        });
        // A previous confirmation describes the saved values, not new edits.
        controls.addEventListener('input', function () { saveFeedback.textContent = ''; });
        choices.addEventListener('change', function () { saveFeedback.textContent = ''; });

        async function poll() {
            // This endpoint reads the Pi's saved status; polling never contacts
            // the NAS. Only explicit button handlers send commands.
            await refresh();
            schedule(poll, 5000);
        }
        return poll();
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = {formatStatus, mount};
    if (root.document) {
        const panel = root.document.getElementById('syncapi-run-panel');
        if (panel) mount(panel, root.document, root.fetch.bind(root), root.setTimeout.bind(root));
    }
})(typeof window === 'undefined' ? globalThis : window);

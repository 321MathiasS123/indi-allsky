(function (root) {
    'use strict';

    function formatStatus(state) {
        const parts = [state.message || state.state || 'Idle'];
        if (typeof state.completed === 'number') {
            parts.push(`${state.completed} of ${state.total} items completed; ${state.skipped} skipped; ${state.files} files, ${(state.bytes / 1048576).toFixed(1)} MiB sent.`);
        }
        if (state.cutoff) parts.push(`Includes completed files through ${state.cutoff.replace('T', ' ')}.`);
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
        let state = {};
        let busy = false;
        let initialized = false;

        function render(value) {
            state = value;
            if (!initialized && Array.isArray(value.types)) {
                value.types.forEach(function (type) {
                    const label = document.createElement('label');
                    const input = document.createElement('input');
                    input.type = 'checkbox';
                    input.value = type.id;
                    input.checked = type.selected;
                    label.appendChild(input);
                    label.appendChild(document.createTextNode(' ' + type.label));
                    choices.appendChild(label);
                });
                initialized = true;
            }
            start.disabled = busy || !value.enabled || value.active;
            cancel.disabled = busy || !value.active || value.cancel_requested;
            choices.disabled = busy || value.active;
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

        async function action(payload) {
            if (busy) return;
            busy = true;
            render(state);
            error.textContent = '';
            try {
                state = await request(payload);
            } catch (exception) {
                error.textContent = exception.message;
            } finally {
                busy = false;
                render(state);
            }
        }

        start.addEventListener('click', function () {
            const types = Array.from(choices.querySelectorAll('input:checked'), input => input.value);
            if (!types.length) {
                error.textContent = 'Select at least one media type.';
                return;
            }
            return action({action: 'start', types: types});
        });
        cancel.addEventListener('click', function () {
            return action({action: 'cancel', task_id: state.task_id});
        });

        async function poll() {
            if (!busy) {
                busy = true;
                try {
                    render(await request());
                } catch (exception) {
                    error.textContent = exception.message;
                } finally {
                    busy = false;
                    render(state);
                }
            }
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

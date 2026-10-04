'use strict';

function initCustomKeogram(options) {
    const form = document.getElementById('custom-keogram-form');
    const fields = document.getElementById('custom-keogram-fields');
    const start = document.getElementById('keogram-start');
    const end = document.getElementById('keogram-end');
    const status = document.getElementById('custom-keogram-status');
    const result = document.getElementById('custom-keogram-result');
    const image = document.getElementById('custom-keogram-image');
    let timer;

    function message(text, error) {
        status.textContent = text;
        status.classList.toggle('tw:text-error', !!error);
    }

    function validate() {
        end.setCustomValidity(end.value && start.value && end.value <= start.value
            ? 'The end must be later than the start.' : '');
    }
    start.addEventListener('input', validate);
    end.addEventListener('input', validate);
    form.addEventListener('invalid', function (event) {
        message(event.target.validationMessage, true);
    }, true);

    async function jsonRequest(url, settings) {
        const response = await fetch(url, Object.assign({signal: AbortSignal.timeout(15000)}, settings));
        if (response.redirected) {
            const error = new Error('Your session has expired. Reload this page and sign in again.');
            error.terminal = true;
            throw error;
        }
        const data = await response.json();
        if (!response.ok) {
            const error = new Error(data.message || 'The request failed. Please try again.');
            error.terminal = response.status >= 400 && response.status < 500 && response.status !== 429;
            throw error;
        }
        return data;
    }

    async function poll(taskId) {
        let retry = false;
        try {
            const url = new URL(options.endpoint, window.location.href);
            url.searchParams.set('task_id', taskId);
            url.searchParams.set('camera_id', options.cameraId);
            const data = await jsonRequest(url);
            if (!data || !['MANUAL', 'QUEUED', 'RUNNING', 'SUCCESS', 'FAILED', 'EXPIRED'].includes(data.state)) {
                throw new Error('Invalid status response.');
            }
            start.value = data.start;
            end.value = data.end;
            if (data.state === 'SUCCESS') {
                if (!data.image_url) throw new Error('Missing keogram preview.');
                image.onload = function () { result.hidden = false; };
                image.onerror = function () {
                    message('The preview could not be loaded. Reload this page to try again.', true);
                };
                image.src = data.image_url;
                document.getElementById('custom-keogram-open').href = data.image_url;
                document.getElementById('custom-keogram-download').href = data.image_url + '&download=1';
                message(data.message + ' Images: ' + data.first + ' to ' + data.last + '.' +
                    (data.resized ? ' ' + data.resized + ' images resized to match the first frame.' : ''), false);
            } else if (data.state === 'FAILED' || data.state === 'EXPIRED') {
                message(data.state === 'EXPIRED' ? 'This job expired. Generate the keogram again.' : data.message, true);
            } else {
                retry = true;
                message(data.state === 'RUNNING'
                    ? 'Creating keogram: ' + ((data.frames || 0) + (data.skipped || 0)) + ' / ' + data.total + ' images checked.'
                    : 'Queued. Generation will start when the image worker is available.', false);
            }
        } catch (error) {
            retry = !error.terminal;
            message(retry ? 'Could not read progress. Retrying automatically…' : error.message, true);
        } finally {
            fields.disabled = retry || !options.enabled;
            if (retry) timer = setTimeout(function () { poll(taskId); }, 3000);
        }
    }

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        if (!options.enabled || fields.disabled) return;
        validate();
        if (!form.reportValidity()) return;
        clearTimeout(timer);
        fields.disabled = true;
        result.hidden = true;
        message('Submitting keogram…', false);
        try {
            const data = await jsonRequest(options.endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json', 'X-CSRFToken': options.csrfToken},
                body: JSON.stringify({camera_id: options.cameraId, start: start.value, end: end.value})
            });
            if (!Number.isInteger(data.task_id) || data.task_id <= 0) throw new Error('Invalid generation response.');
            const url = new URL(window.location.href);
            url.searchParams.set('task_id', data.task_id);
            url.searchParams.set('camera_id', options.cameraId);
            window.history.replaceState(null, '', url);
            await poll(data.task_id);
        } catch (error) {
            fields.disabled = !options.enabled;
            message(error.message || 'Unable to submit the keogram. Please try again.', true);
        }
    });

    const params = new URL(window.location.href).searchParams;
    const taskId = Number(params.get('task_id'));
    if (options.enabled && Number.isInteger(taskId) && taskId > 0 && Number(params.get('camera_id')) === options.cameraId) {
        fields.disabled = true;
        poll(taskId);
    }
}

if (typeof module !== 'undefined') module.exports = {initCustomKeogram};

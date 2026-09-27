// Also run through pytest by test_gallery_panorama.py.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('panorama toolbar follows the selected slide and clears unavailable links', () => {
    const template = fs.readFileSync(path.join(__dirname,
        '../../indi_allsky/flask/templates/gallery.html'), 'utf8');
    const start = template.indexOf('function populate_gallery(');
    const end = template.indexOf('\nfunction ', start + 1);
    const elements = [];
    const controls = new Map();
    const changes = [];
    const pswp = {
        ui: { registerElement(options) {
            const el = {
                style: {},
                setAttribute(key, value) { this[key] = value; },
                removeAttribute(key) { delete this[key]; },
            };
            controls.set(options.name, el);
            if (['panorama-button', 'download-button'].includes(options.name)) {
                options.onInit(el, pswp);
            }
        } },
        on(event, callback) { if (event === 'change') changes.push(callback); },
    };
    const context = {
        asi676mc_repair_gallery_enabled: false,
        scrollToTop() {},
        PhotoSwipe: {},
        lightbox: null,
        $(selector, attrs) {
            if (selector === '<a />') elements.push(attrs);
            return { empty() {}, appendTo() {} };
        },
        PhotoSwipeLightbox: class {
            constructor(options) { this.options = options; this.pswp = pswp; }
            on(event, callback) { if (event === 'uiRegister') this.register = callback; }
            init() { this.register(); }
        },
    };
    vm.createContext(context);
    vm.runInContext(template.slice(start, end).replace(/{{\s*url_for\('([^']+)'\)\s*}}/g,
        (_, route) => `/indi-allsky/${route.split('.').pop()}`), context);
    context.populate_gallery([
        { id: 1, panorama_id: 21, url: 'first.jpg' },
        { id: 2, panorama_id: null, url: 'second.jpg' },
        { id: 3, panorama_id: 23, url: 'third.jpg' },
    ]);
    const panorama = controls.get('panorama-button');
    const download = controls.get('download-button');
    const select = (index) => {
        pswp.currSlide = { data: { src: elements[index].href, element: {
            getAttribute(key) { return String(elements[index][key] ?? ''); },
        } } };
        changes.forEach(callback => callback());
    };
    select(0);
    assert.equal(panorama.href, '/indi-allsky/panorama_image_view?id=21');
    assert.equal(panorama.style.display, '');
    assert.equal(panorama.target, '_blank');
    assert.equal(panorama.rel, 'noopener');
    assert.equal(download.href, 'first.jpg');
    select(1);
    assert.equal(panorama.style.display, 'none');
    assert.equal(panorama.href, undefined);
    assert.equal(download.href, 'second.jpg');
    select(2);
    assert.equal(panorama.style.display, '');
    assert.equal(panorama.href, '/indi-allsky/panorama_image_view?id=23');
    assert.equal(download.href, 'third.jpg');
    assert.equal(context.lightbox.options.padding.top, 60);
});

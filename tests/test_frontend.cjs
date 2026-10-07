const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this.style = {}; this.value = ''; this.hidden = false; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  get firstElementChild() { return this.children[0]; }
}
const listeners = {};
const app = { registerExtension(extension) { this.extension = extension; } };
const api = { apiURL: path => 'http://localhost:8188' + path, addEventListener(name, callback) { listeners[name] = callback; } };
const ComfyWidgets = { STRING(node, name) { const widget = { name, inputEl: new Element('textarea'), options: {} }; node.widgets.push(widget); return { widget }; } };
let source = fs.readFileSync(require('node:path').join(__dirname, '../web/fullstack.js'), 'utf8');
source = source.replace(/^import .*;\r?\n/gm, '').replace('import.meta.url', "'http://localhost:8188/extensions/h3fs/fullstack.js'");
const context = vm.createContext({ app, api, ComfyWidgets, console, URL, URLSearchParams, document: { createElement: tag => new Element(tag), head: new Element('head') } });
vm.runInContext(source + '\nthis.subject={Panel,panels,panelFor,promptOwner,showText,completedOutput};', context);
const { Panel, panels, panelFor, promptOwner, showText, completedOutput } = context.subject;

function fixture() {
  const graph = { id: 'workflow-a', links: {}, getNodeById(id) { return this.nodes[id]; }, setDirtyCanvas() {}, nodes: {} };
  const node = { id: 1, type: 'H3FullStack', properties: {}, widgets: [{ name: 'prompt', value: 'An idea' }], graph, outputs: [{}, {}, { type: 'STRING', links: [10] }] };
  const textNode = { id: 9, type: 'ShowText|pysssss', properties: {}, graph, inputs: [{ name: 'text', link: 10 }], widgets: [], size: [220, 80], setSize(size) { this.size = size; } };
  graph.nodes = { 1: node, 9: textNode }; graph.links[10] = { origin_id: 1, origin_slot: 2, target_id: 9 };
  const panel = Object.create(Panel.prototype);
  Object.assign(panel, { node, state: { preview: true }, controls: { optimize_on_run: new Element('input'), generate_after_optimization: new Element('input') }, prompt: new Element('textarea'), reviewButton: new Element('button'), optimizedPrompt: new Element('textarea'), videoChoice: new Element('select'), preview: new Element('div'), statLabels: new Element('div'), save() { this.saved = JSON.stringify(this.state); }, update() {} });
  node.h3fsPanel = panel;
  return { graph, node, textNode, panel };
}
function historyFixture() {
  return { success: { prompt: [1, 'success', {
    1: { class_type: 'H3FullStack', inputs: { prompt: 'An idea', settings: '{"optimize_on_run":true}' } }, 7: { inputs: { video: ['1', 0] } }, 8: { inputs: { video: ['1', 1] } }, 9: { inputs: { text: ['1', 2] } },
  }, { extra_pnginfo: { workflow: { id: 'workflow-a' } } }], status: { completed: true, status_str: 'success' }, outputs: {
    7: { images: [{ filename: 'base.mp4', subfolder: 'video', type: 'output' }] }, 8: { images: [{ filename: 'upscaled.mp4', subfolder: 'video', type: 'output' }] }, 9: { text: ['A ceramic cup on a table.'] },
  } } };
}

test('optimized prompt fills the panel and the connected native STRING display', () => {
  const { node, textNode, panel } = fixture();
  panel.setPrompt('A ceramic cup on a table.');
  assert.equal(panel.optimizedPrompt.value, node.properties.h3fs_prompt);
  assert.equal(textNode.widgets[0].value, node.properties.h3fs_prompt);
  assert.equal(textNode.widgets[0].inputEl.readOnly, true);
  assert.equal(textNode.widgets[0].options.serialize, false);
  assert.equal(node.outputs[2].type, 'STRING');
});

test('Chinese checkbox persists its value and invalidates automatic reviewed-prompt reuse', () => {
  const { panel } = fixture(); const parent = new Element('div');
  panel.state.use_reviewed_prompt = true;
  const checkbox = panel.check(parent, 'use_chinese_prompt', '使用中文提示词', () => { panel.invalidateReviewed(); panel.save(); });
  assert.equal(checkbox.attributes['aria-label'], '使用中文提示词');
  assert.equal(checkbox.checked, false);
  checkbox.checked = true; checkbox.onchange();
  assert.equal(JSON.parse(panel.saved).use_chinese_prompt, true);
  assert.equal(JSON.parse(panel.saved).use_reviewed_prompt, false);
  checkbox.checked = false; checkbox.onchange();
  assert.equal(JSON.parse(panel.saved).use_chinese_prompt, false);
});
test('Show Text compatibility reuses one widget and ignores non-H3 sources', () => {
  const { textNode, node } = fixture(); showText(textNode, 'first'); showText(textNode, 'second');
  assert.equal(textNode.widgets.length, 1); assert.equal(textNode.widgets[0].value, 'second');
  assert.equal(promptOwner(textNode), node); node.type = 'OtherText'; assert.equal(promptOwner(textNode), null);
});
test('completed video has native controls and is not autoplaying or looping', () => {
  const { panel } = fixture(); panel.renderResults([{ url: '/h3fs/file/base' }]); const video = panel.preview.firstElementChild;
  assert.equal(video.tagName, 'VIDEO'); assert.equal(video.controls, true); assert.equal(video.preload, 'metadata');
  assert.notEqual(video.autoplay, true); assert.notEqual(video.loop, true); assert.equal(video.src, 'http://localhost:8188/h3fs/file/base');
});
test('upscaled selection swaps the same player area', () => {
  const { panel } = fixture(); panel.renderResults([{ url: '/base' }, { url: '/upscaled' }]);
  assert.equal(panel.videoChoice.hidden, false); panel.videoChoice.value = '1'; panel.videoChoice.onchange();
  assert.equal(panel.preview.children.length, 1); assert.equal(panel.preview.firstElementChild.src, 'http://localhost:8188/upscaled');
});
test('late sampling frames cannot overwrite the completed player; a new run resets it', () => {
  const { panel } = fixture(); panel.renderResults([{ url: '/base' }]); const video = panel.preview.firstElementChild;
  panel.previewEvent({ image: 'late', step: 4 }); assert.equal(panel.preview.firstElementChild, video);
  panel.startPreview(); panel.previewEvent({ image: 'next' }); assert.equal(panel.preview.firstElementChild.tagName, 'IMG');
});
test('turning off live sampling does not suppress a completed video', () => {
  const { panel } = fixture(); panel.state.preview = false; panel.renderResults([{ url: '/base' }]); assert.equal(panel.preview.hidden, false);
  panel.startPreview(); assert.equal(panel.preview.hidden, true);
});
test('history restores prompt and both videos from actual downstream outputs', () => {
  const { node } = fixture(); const output = completedOutput(historyFixture(), node);
  assert.equal(output.text, 'A ceramic cup on a table.'); assert.equal(output.files.length, 2);
  assert.match(output.files[0].url, /filename=base.mp4/); assert.match(output.files[1].url, /filename=upscaled.mp4/);
});
test('history never restores a different workflow with the same node ID', () => {
  const { node } = fixture(); node.graph.id = 'workflow-b'; assert.equal(completedOutput(historyFixture(), node), undefined);
});
test('failed or incomplete execution does not replace completed results', () => {
  const { node } = fixture(); const history = historyFixture(); history.success.status.completed = false;
  assert.equal(completedOutput(history, node), undefined); history.success.status.completed = true; history.success.status.status_str = 'error';
  assert.equal(completedOutput(history, node), undefined);
});
test('events select the active graph rather than a stale panel with the same ID', () => {
  const a = fixture(), b = fixture(); panels.clear(); panels.set(a.node, a.panel); panels.set(b.node, b.panel); app.graph = b.graph;
  assert.equal(panelFor('1'), b.panel); assert.equal(panelFor('1.4'), b.panel);
});
test('Show Text executed hook displays backend UI text without another model run', async () => {
  const { textNode, panel } = fixture(); function ShowText() {};
  await app.extension.beforeRegisterNodeDef(ShowText, { name: 'ShowText|pysssss' });
  ShowText.prototype.onExecuted.call(textNode, { text: ['A quiet kitchen.'] });
  assert.equal(textNode.widgets[0].value, 'A quiet kitchen.'); assert.equal(panel.optimizedPrompt.value, 'A quiet kitchen.');
});

test('turning optimization off reuses the optimized prompt for the unchanged intent', () => {
  const { panel } = fixture(); panel.setPrompt('Reviewed result', { original: 'An idea', optimized: true });
  panel.optimizationChanged();
  assert.equal(panel.state.use_reviewed_prompt, true); assert.equal(panel.state.reviewed_prompt, 'Reviewed result');
  assert.equal(panel.state.optimize_on_run, false); assert.equal(panel.state.generate_after_optimization, true);
  assert.equal(panel.reviewButton.textContent, 'Use Input Prompt');
});
test('changed intent or media cannot silently reuse an old prompt', () => {
  const { panel, node } = fixture(); panel.setPrompt('Reviewed result', { original: 'An idea', optimized: true });
  node.widgets[0].value = 'A different idea'; panel.optimizationChanged(); assert.notEqual(panel.state.use_reviewed_prompt, true);
  node.widgets[0].value = 'An idea'; panel.invalidateReviewed(); panel.optimizationChanged(); assert.equal(panel.state.use_reviewed_prompt, false);
  assert.equal(panel.optimizedPrompt.value, 'Reviewed result');
});
test('explicit review overrides connected text without disconnecting it; reenable optimizer persists', () => {
  const { panel, node } = fixture(); node.inputs = [{ name: 'prompt_external', link: 99 }]; panel.prompt.disabled = true;
  panel.setPrompt('Reviewed external result'); panel.selectReviewedPrompt();
  assert.equal(node.inputs[0].link, 99); assert.equal(panel.state.reviewed_prompt, 'Reviewed external result');
  panel.state.optimize_on_run = true; panel.optimizationChanged();
  assert.equal(JSON.parse(panel.saved).use_reviewed_prompt, false);
});
test('optimize-only history restores parent UI text without any downstream display', () => {
  const { node } = fixture(); const history = historyFixture(); history.success.outputs = { 1: { text: ['Reviewed result'] } };
  const result = completedOutput(history, node); assert.equal(result.text, 'Reviewed result');
  assert.equal(result.original, 'An idea'); assert.equal(result.optimized, true); assert.equal(result.files.length, 0);
});

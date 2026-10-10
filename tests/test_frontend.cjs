const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this.style = {setProperty(){}}; this.value = ''; this.hidden = false; this.dataset={};this.classList={toggle(){},add(){}}; }
  append(...children) { for(const c of children){if(c.parent)c.parent.children=c.parent.children.filter(x=>x!==c);c.parent=this;this.children.push(c);} }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  closest(tag){let p=this;while(p&&p.tagName!==tag.toUpperCase())p=p.parent;return p;}
  querySelector(selector){return this.querySelectorAll(selector)[0];}
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  removeAttribute(key) { delete this.attributes[key]; }
  showModal() {this.open=true;}
  close() {this.open=false;}
  remove() {this.removed=true;}
  addEventListener() {}
  prepend(...children) { for(const c of [...children].reverse()){if(c.parent)c.parent.children=c.parent.children.filter(x=>x!==c);c.parent=this;this.children.unshift(c);} }
  checkValidity() {return true;}
  querySelectorAll(selector) {const result=[];const classes=selector.split(',');const visit=n=>{for(const child of n.children||[]){if(classes.some(c=>c.startsWith('.')?child.className?.split(' ').includes(c.slice(1)):child.tagName===c.toUpperCase()))result.push(child);visit(child);}};visit(this);return result;}
  get firstElementChild() { return this.children[0]; }
}
const listeners = {};
const app = { registerExtension(extension) { if(!this.extension||extension.name==='EverythingIsParty.H3FullStack')this.extension = extension; } };
const api = { apiURL: path => 'http://localhost:8188' + path, addEventListener(name, callback) { listeners[name] = callback; } };
const ComfyWidgets = { STRING(node, name) { const widget = { name, inputEl: new Element('textarea'), options: {} }; node.widgets.push(widget); return { widget }; } };
let source = fs.readFileSync(require('node:path').join(__dirname, '../web/fullstack.js'), 'utf8');
source = source.replace(/^import .*;\r?\n/gm, '').replace('import.meta.url', "'http://localhost:8188/extensions/h3fs/fullstack.js'");
class FormDataFixture { constructor(){this.entries=[];} append(...entry){this.entries.push(entry);} }
const context = vm.createContext({ app, api, ComfyWidgets, console, URL, URLSearchParams, FormData:FormDataFixture, structuredClone, window:{addEventListener(){}}, requestAnimationFrame:fn=>fn(),document: { createElement: tag => new Element(tag),createElementNS:(_,tag)=>new Element(tag), head: new Element('head'),body:new Element('body') } });
vm.runInContext(source + '\nthis.subject={Panel,panels,panelFor,promptOwner,showText,completedOutput,migrate,translated,mountPosition,bindCornerInputs,publishCornerLayout};', context);
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

test('mounting a second generator places it beside the host without covering existing nodes',()=>{
  const host={pos:[530,50],size:[600,940],graph:{_nodes:[]}},source={pos:[0,0],size:[450,780]},first={pos:[20,50],size:[450,780]};
  host.graph._nodes=[host,first,source];const pos=context.subject.mountPosition(host,source,380);
  assert.equal(pos[0],0);assert.ok(pos[1]>=first.pos[1]+first.size[1]+48);
});
test('both native and Vue geometry bind links to the visible image corner, not the header',()=>{
  class Node{getInputPos(){return [10,20];}getInputSlotPos(){return [10,20];}getConnectionPos(){return [10,20];}}
  context.subject.bindCornerInputs(Node);const n=new Node();n.pos=[530,50];n.inputs=[{name:'image_1'}];n.flags={};n.h3fsPanel={updatePort(){n.inputs[0].pos=[24,380];}};
  assert.deepEqual([...n.getInputPos(0)],[554,430]);assert.deepEqual([...n.getInputSlotPos(0)],[554,430]);
  const out=new Float32Array(2);assert.equal(n.getConnectionPos(true,0,out),out);assert.deepEqual([...out],[554,430]);
  assert.deepEqual(n.getConnectionPos(false,0),[10,20]);n.flags.collapsed=true;assert.deepEqual(n.getInputPos(0),[10,20]);
});
test('Vue/canvas fallback link registry receives scoped actual corner offsets',()=>{
  let call;const registry={updateNodeSlotOffsets(...args){call=args;}},node={id:1,graph:{rootGraph:{id:'workflow-a'}},flags:{},inputs:[{pos:[24,380]},{pos:[210,380]}],outputs:[]};
  context.subject.publishCornerLayout({node},registry);assert.equal(call[0],'workflow-a');assert.equal(call[1],1);assert.equal(call[3],'expanded');
  assert.deepEqual(JSON.parse(JSON.stringify(call[2])),[{type:'input',index:0,position:{x:24,y:380}},{type:'input',index:1,position:{x:210,y:380}}]);
  node.flags.collapsed=true;call=undefined;context.subject.publishCornerLayout({node},registry);assert.equal(call,undefined);
});

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
test('old workflow enables auto context by default and the accessible switch preserves the saved minimum', () => {
  const { panel, node } = fixture();
  node.widgets.push({ name: 'settings', value: '{"version":2,"llm_ctx":8192}' });
  panel.cfg = { defaults: { version: 2, media: [], llm_ctx: 8192, llm_auto_context: true, llm_mmproj: 'projector.gguf' } };
  panel.readState();
  assert.equal(panel.state.llm_auto_context, true);
  const checkbox = panel.check(new Element('div'), 'llm_auto_context', 'Auto Context Length');
  assert.equal(checkbox.checked, true);
  assert.equal(checkbox.attributes['aria-label'], 'Auto Context Length');
  checkbox.checked = false; checkbox.onchange();
  assert.equal(JSON.parse(panel.saved).llm_auto_context, false);
  assert.equal(JSON.parse(panel.saved).llm_ctx, 8192);
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

test('history restores the built-in MP4 without an external Save Video node',()=>{
  const {node}=fixture(),history=historyFixture();history.success.prompt[2]={1:history.success.prompt[2][1]};
  history.success.outputs={1:{text:['old intermediate'],h3fs_files:[{url:'/actual-final.mp4',name:'Original'}],h3fs_final_prompt:['完整视频提示词']}};
  const result=completedOutput(history,node);assert.equal(result.files[0].url,'/actual-final.mp4');assert.equal(result.text,'完整视频提示词');
});

test('real expanded-node history restores the parent video and not another node subtree',()=>{
  const {node}=fixture(),history=historyFixture();history.success.prompt[2]={1:history.success.prompt[2][1]};
  history.success.outputs={'1.0.0.2':{text:['完整提示词']},'1.0.0.2.0.0.19':{h3fs_files:[{url:'/actual.mp4'}],h3fs_final_prompt:['完整提示词']},'10.0.0':{h3fs_files:[{url:'/wrong.mp4'}]}};
  const result=completedOutput(history,node);assert.equal(result.files[0].url,'/actual.mp4');assert.equal(result.text,'完整提示词');
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

test('v2 migration defaults UI and prompts to Chinese without discarding stored media', () => {
  const old = { version: 2, use_chinese_prompt: false, use_reviewed_prompt: true, resolution: 'megapixels', media: [{ kind:'image',slot:0,path:'kept.png' }] };
  const result=context.subject.migrate(old);assert.equal(result.ui_language,'zh');assert.equal(result.use_chinese_prompt,true);assert.equal(result.use_reviewed_prompt,false);assert.equal(result.media[0].path,'kept.png');
  const english=context.subject.migrate({version:3,ui_language:'en',use_chinese_prompt:false,media:[],reference_cards:[]});assert.equal(english.ui_language,'en');assert.equal(english.use_chinese_prompt,false);
});
test('common labels localize without changing protocol or model names', () => {
  assert.equal(context.subject.translated('Duration (Seconds)','zh'),'时长（秒）');
  assert.equal(context.subject.translated('Image 9 Input','zh'),'图片 9 接入口');
  assert.equal(context.subject.translated('qwen_image_2.1_bf16.safetensors','zh'),'qwen_image_2.1_bf16.safetensors');
  assert.equal(context.subject.translated('REF','zh'),'REF');
  assert.equal(context.subject.translated('使用中文提示词','en'),'Use Chinese Prompts');
});
test('each image slot owns a separate editable draft and aspect', () => {
  const { panel }=fixture();panel.state.reference_cards=[];
  const a=panel.referenceCard(0),b=panel.referenceCard(8);a.text='杯子';a.draft='陶瓷杯';a.aspect='1:1';b.text='房间';
  assert.equal(panel.referenceCard(0),a);assert.equal(b.draft,'');assert.equal(b.aspect,'video');
  panel.clearDrafts();assert.equal(a.draft,'');assert.equal(a.text,'杯子');assert.equal(b.text,'房间');
});

function renderFixture(type='H3FullStack',enabled=false){
  const {node,graph}=fixture();graph.change=()=>{};node.type=type;node.widgets.push({name:'settings',value:'{}'},{name:'seed',value:42},{name:'intent',value:'居家 VLOG'},{name:'action',value:'design'});node.inputs=[];node.flags={};node.addInput=(name,type)=>node.inputs.push({name,type});
  const defaults={version:3,ui_language:'zh',use_chinese_prompt:true,mode:'auto',media:[],loras:[],aspect:'16:9',megapixels:1,multiple:32,preview:true,llm_mmproj:'None',auto_reference:enabled,reference_cards:[],image_loras:[],style_text:'',reference_job:'',image_engine:'qwen21',image_steps:25};
  const cfg={defaults,models:{LLM:[],diffusion_models:[],text_encoders:[],vae:[],vae_approx:[],loras:[]}};
  return new Panel(node,cfg);
}
test('main panel stays a simple nine-grid even when loading old automatic settings',()=>{
  const normal=renderFixture();assert.equal(normal.mediaList.querySelectorAll('.h3-reference-editor').length,0);
  const automatic=renderFixture('H3FullStack',true);assert.equal(automatic.mediaList.querySelectorAll('.h3-reference-editor').length,0);assert.equal(automatic.mediaList.querySelectorAll('.h3-object-name').length,0);assert.equal(automatic.state.auto_reference,false);assert.equal(automatic.referenceProgressBar,undefined);assert.equal(automatic.mediaList.querySelectorAll('.h3-image-port').length,9);assert.equal(automatic.root.querySelectorAll('.h3-reference-entry').length,0);
});
test('standalone assistant renders without video controls and retains nine result outputs',()=>{
  const assistant=renderFixture('H3ReferenceAssistant');assert.equal(assistant.mediaList.querySelectorAll('.h3-reference-editor').length,0);assert.equal(assistant.mediaList.querySelectorAll('.h3-object-name').length,9);assistant.openImageSettings();assert.ok(assistant.controls.image_model);assert.ok(assistant.controls.llm_model);assert.equal(assistant.preview,undefined);
});

test('assistant has nine per-card IMAGE output corners and an external intent input',()=>{
  const p=renderFixture('H3ReferenceAssistant');assert.equal(p.mediaList.querySelectorAll('.h3-output-port').length,9);
  assert.equal(p.portButtons.has('intent'),true);assert.equal(p.outputButtons.size,9);
  assert.equal(context.subject.translated('Image 9 Output','zh'),'图片 9 输出口');
  p.node.inputs.find(x=>x.name==='intent').link=10;p.update();assert.equal(p.prompt.disabled,true);
});

test('assistant standard-run stage selector changes the native serialized action widget',()=>{
  const p=renderFixture('H3ReferenceAssistant');const select=p.root.querySelectorAll('select').find(x=>x.attributes['aria-label']==='运行阶段');
  assert.equal(select.value,'design');select.value='all';select.onchange();assert.equal(p.node.widgets.find(x=>x.name==='action').value,'all');
});

test('stepwise reference queue includes only external text/image ancestors, never video outputs',async()=>{
  const p=renderFixture('H3FullStack',true);p.ensureJob=()=> 'a'.repeat(32);p.node.graph.serialize=()=>({id:'workflow-a'});
  p.node.inputs.find(x=>x.name==='prompt_external').link=11;p.node.inputs.find(x=>x.name==='image_1').link=12;
  p.referenceCard(0).enabled=true;p.referenceCard(1).enabled=true;p.referenceCard(1).text='杯子';
  app.graphToPrompt=async()=>({workflow:{id:'workflow-a'},output:{
    1:{class_type:'H3FullStack',inputs:{prompt_external:['4',0],image_1:['5',0]}},
    4:{class_type:'CR Prompt Text',inputs:{prompt:['6',0]}},5:{class_type:'LoadImage',inputs:{image:'kept.png'}},
    6:{class_type:'PrimitiveStringMultiline',inputs:{value:'实际外接意图'}},7:{class_type:'SaveVideo',inputs:{video:['1',0]}}
  }});
  let sent;api.queuePrompt=async(_,data)=>sent=data;await p.queueReference('design');
  assert.deepEqual(Object.keys(sent.output).sort(),['4','5','6','reference_studio']);
  assert.equal(sent.output.reference_studio.inputs.intent[0],'4');assert.equal(sent.output.reference_studio.inputs.image_1[0],'5');
  const cards=JSON.parse(sent.output.reference_studio.inputs.settings).reference_cards;assert.equal(cards[0].enabled,false);assert.equal(cards[1].enabled,true);
  assert.equal(p.referenceCard(0).enabled,true);assert.equal(p.referenceBusy,false);
});

test('I2V stepwise actions filter inactive slots without clearing saved inputs',async()=>{
  const p=renderFixture('H3FullStack',true);p.ensureJob=()=> 'b'.repeat(32);p.node.graph.serialize=()=>({});app.graph=p.node.graph;p.state.mode='first_last';
  for(let i=0;i<9;i++){p.referenceCard(i).enabled=true;p.referenceCard(i).text='原文 '+i;}
  let sent;api.queuePrompt=async(_,data)=>sent=data;await p.queueReference('design');
  assert.ok(sent,p.statusEl.textContent);const cards=JSON.parse(sent.output.reference_studio.inputs.settings).reference_cards;assert.equal(cards.filter(x=>x.enabled).length,2);
  assert.equal(p.state.reference_cards.filter(x=>x.enabled).length,9);assert.equal(p.state.reference_cards[8].text,'原文 8');
});

test('T2V refuses reference generation without modifying stored image prompts',async()=>{
  const p=renderFixture('H3FullStack',true);p.state.mode='text';p.referenceCard(0).text='保留';let calls=0;api.queuePrompt=async()=>calls++;
  await p.queueReference('all');assert.equal(calls,0);assert.equal(p.referenceCard(0).text,'保留');assert.match(p.statusEl.textContent,/T2V/);assert.equal(p.referenceBusy,false);
});

test('output corner updates both connection coordinates and the current native renderer bounds',()=>{
  const {panel:p,node}=fixture();node.flags={};node.pos=[100,200];node.outputs[2].name='图片 1';node.outputs[2].boundingRect=[0,0,0,0];
  p.outputButtons=new Map([[2,{isConnected:true,getBoundingClientRect:()=>({left:400,top:300,width:20,height:20})}]]);
  p.root={getBoundingClientRect:()=>({top:0,bottom:700})};app.canvas={canvas:{getBoundingClientRect:()=>({left:0,top:0})},ds:{scale:1,offset:[0,0]}};
  p.updateOutput(2);assert.equal(node.outputs[2].pos[0],310);assert.equal(node.outputs[2].pos[1],110);
  assert.equal(node.outputs[2].boundingRect[0],400);assert.equal(node.outputs[2].label,' ');assert.equal(node.outputs[2].name,'图片 1');
});

test('selecting a card shows one editor without losing the other eight prompts',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.referenceCard(0).text='美女';p.referenceCard(1).text='小院子';p.referenceCard(1).draft='无人小院子';p.renderMedia();
  const selects=p.mediaList.querySelectorAll('.h3-card-select');selects[1].onclick();
  assert.equal(p.mediaList.querySelectorAll('.h3-reference-editor').length,0);assert.equal(p.dialog.querySelectorAll('.h3-reference-editor').length,1);
  const inputs=p.dialog.querySelectorAll('textarea');assert.equal(inputs[0].value,'小院子');assert.equal(inputs[1].value,'无人小院子');assert.equal(p.referenceCard(0).text,'美女');
});

test('model and LoRA settings have a direct visible entry and open outside the canvas',()=>{
  const p=renderFixture('H3ReferenceAssistant');const entry=p.root.querySelectorAll('.h3-reference-entry')[0];assert.ok(entry);
  entry.children[0].onclick();assert.equal(p.dialog.open,true);assert.equal(p.dialog.attributes['aria-label'],'生图模型 / LoRA');assert.ok(p.controls.image_steps);assert.ok(p.controls.image_model);
});

test('existing thumbnail opens an original-image viewer, not the file picker',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.state.media=[{kind:'image',slot:0,path:'a.png',preview_url:'/h3fs/file/safe'}];p.renderMedia();
  p.mediaList.querySelectorAll('img')[0].onclick();const img=p.dialog.querySelectorAll('img')[0];assert.equal(img.src,'http://localhost:8188/h3fs/file/safe');assert.equal(img.className,'h3-full-image');
});

test('legacy drafts are invalidated, but generated image files and object names survive',()=>{
  const old={version:3,reference_cards:[{slot:1,text:'小院子',draft:'美女坐在小院子'}],media:[{kind:'image',slot:1,path:'kept.png'}]};
  context.subject.migrate(old);assert.equal(old.reference_cards[0].draft,'');assert.equal(old.reference_cards[0].text,'小院子');assert.equal(old.media[0].path,'kept.png');
});

test('restoring an old job cannot put defective prompts back into the editor',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.referenceCard(0).text='蛋糕';p.referenceResult({id:'a'.repeat(32),intent:'居家 VLOG',cards:{0:{slot:0,text:'蛋糕',prompt:'美女拿着蛋糕',path:'kept.png'}}},false);
  assert.equal(p.referenceCard(0).draft,'');assert.equal(p.state.media[0].path,'kept.png');
});

test('shared person and scene reference controls persist independently in the content dialog',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.state.style_image='person.png';p.state.scene_reference='scene.png';
  p.referenceCard(0).text='美女';p.referenceCard(0).draft='old prompt';p.openStyleSettings();
  assert.equal(p.dialog.attributes['aria-label'],'参考学习');
  const zones=p.dialog.querySelectorAll('.h3-reference-drop');assert.equal(zones.length,1);
  assert.equal(zones[0].attributes['aria-label'],'画风 / 人物参考图');
  const strict=p.controls.strict_person_reference;strict.checked=true;strict.onchange();
  const saved=JSON.parse(p.node.widgets.find(x=>x.name==='settings').value);
  assert.equal(saved.strict_person_reference,true);assert.equal(saved.style_image,'person.png');assert.equal(saved.scene_reference,'scene.png');assert.equal(saved.reference_cards[0].draft,'');
  p.openStyleSettings('scene');assert.equal(p.dialog.querySelectorAll('.h3-reference-drop')[0].attributes['aria-label'],'场景参考（只作用场景素材）');
});

test('dropping a per-card content image enables its own strict edit without changing other slots',async()=>{
  const p=renderFixture('H3ReferenceAssistant');p.referenceCard(0).draft='old';p.referenceCard(1).text='小院子';
  p.openCardSettings(0);const zone=p.dialog.querySelectorAll('.h3-reference-drop')[0];let prevented=0,stopped=0,request;
  api.fetchApi=async(path,options)=>{request={path,options};return {ok:true,json:async()=>({kind:'image',path:'h3_fullstack/new-person.png'})};};
  await zone.ondrop({preventDefault(){prevented++;},stopPropagation(){stopped++;},dataTransfer:{files:[{name:'person.png'}]}});
  assert.equal(prevented,1);assert.equal(stopped,1);assert.equal(request.path,'/h3fs/upload');assert.equal(request.options.method,'POST');
  assert.equal(request.options.body.entries[0][0],'file');assert.equal(p.referenceCard(0).reference_image,'h3_fullstack/new-person.png');
  assert.equal(p.referenceCard(0).strict_reference,true);assert.equal(p.referenceCard(0).draft,'');assert.equal(p.referenceCard(1).reference_image,undefined);
  const remove=p.dialog.querySelectorAll('.h3-reference-drop')[0].querySelectorAll('button').find(x=>x.textContent==='移除参考图');remove.onclick();
  assert.equal(p.referenceCard(0).reference_image,'');assert.equal(p.referenceCard(0).strict_reference,false);
});

test('invalid dropped media never replaces a stored content reference',async()=>{
  const p=renderFixture('H3ReferenceAssistant');let changed=0;
  const zone=p.referenceUpload(new Element('div'),'Content Reference','kept.png',()=>changed++);
  api.fetchApi=async()=>({ok:true,json:async()=>({kind:'video',path:'not-an-image.mp4'})});
  await zone.ondrop({preventDefault(){},stopPropagation(){},dataTransfer:{files:[{}]}});
  assert.equal(changed,0);assert.match(p.statusEl.textContent,/请选择图片/);
});

test('nine object fields are directly editable and editing one does not mutate its neighbors',()=>{
  const p=renderFixture('H3ReferenceAssistant');const names=p.mediaList.querySelectorAll('.h3-object-name');
  assert.equal(names.length,9);names[1].value='小院子';names[1].oninput();names[2].value='蛋糕';names[2].oninput();
  assert.equal(p.referenceCard(1).text,'小院子');assert.equal(p.referenceCard(1).enabled,true);assert.equal(p.referenceCard(2).text,'蛋糕');assert.equal(p.referenceCard(0).text,'');
  p.openCardSettings(1);assert.equal(p.dialog.attributes['aria-label'],'图片 2 · 提示词 / 参数');
  const camera=p.dialog.querySelectorAll('input').find(x=>x.attributes['aria-label']==='图片 2 拍摄视角（留空由意图安排）');camera.value='低机位侧面中景';camera.oninput();assert.equal(p.referenceCard(1).camera,'低机位侧面中景');
});

test('learn-reference action can queue before an intent is entered and never includes video branches',async()=>{
  const p=renderFixture('H3ReferenceAssistant');p.node.widgets.find(x=>x.name==='intent').value='';p.ensureJob=()=> 'c'.repeat(32);p.node.graph.serialize=()=>({});
  let sent;api.queuePrompt=async(_,data)=>sent=data;await p.queueReference('analyze');
  assert.equal(sent.output.reference_studio.inputs.action,'analyze');assert.equal(Object.keys(sent.output).length,1);
});

test('completed designs show editable core text separately from the assembled profile prompt',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.referenceCard(0).text='美女';p.openCardSettings(0);
  p.referenceResult({id:'d'.repeat(32),contract_version:4,intent:'居家 VLOG',cards:{0:{slot:0,text:'美女',draft:'在小院散步',prompt:'在小院散步\n人物设定：黑发白衣'}}});
  assert.equal(p.referenceCard(0).draft,'在小院散步');assert.equal(p.cardDraftField.value,'在小院散步');assert.match(p.actualPromptField.value,/人物设定/);
});

test('duplicate enabled LoRAs are visible warnings, never silently removed from saved settings',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.state.image_loras=[{name:'CML.safetensors',weight:1,enabled:true},{name:'CML.safetensors',weight:1,enabled:true}];p.openImageSettings();
  assert.match(p.dialog.querySelectorAll('.h3-error')[0].textContent,/重复叠加/);assert.equal(p.state.image_loras.length,2);
});

test('image LoRA rows display and edit the actual row, not the entries index',()=>{
  const p=renderFixture('H3ReferenceAssistant');p.cfg.models.loras=['CML.safetensors','style.safetensors'];
  p.state.image_loras=[{name:'CML.safetensors',weight:.75,enabled:false},{name:'style.safetensors',weight:.4,enabled:true}];p.openImageSettings();
  let row=p.dialog.querySelectorAll('.h3-lora')[0];
  assert.equal(row.querySelectorAll('select')[0].value,'CML.safetensors');
  assert.equal(row.querySelectorAll('input').find(x=>x.type==='number').value,.75);
  const check=row.querySelectorAll('input').find(x=>x.type==='checkbox');assert.equal(check.checked,false);assert.doesNotMatch(check.attributes['aria-label'],/object Object/);
  check.checked=true;check.onchange();assert.equal(p.state.image_loras[0].enabled,true);
  row=p.dialog.querySelectorAll('.h3-lora')[0];const model=row.querySelectorAll('select')[0];model.value='style.safetensors';model.onchange();
  assert.equal(p.state.image_loras[0].name,'style.safetensors');
  row=p.dialog.querySelectorAll('.h3-lora')[0];const weight=row.querySelectorAll('input').find(x=>x.type==='number');weight.value=.5;weight.oninput();
  assert.equal(p.state.image_loras[0].weight,.5);assert.equal(p.state.image_loras[1].weight,.4);
});

test('generated IMAGE fanout fills empty slots and requires approval for occupied slots',()=>{
  const p=renderFixture(),source={id:3,type:'H3ImageGenerator',properties:{},graph:p.node.graph};p.node.graph.nodes[3]=source;
  for(let slot=0;slot<3;slot++){p.node.inputs[slot].link=100+slot;p.node.graph.links[100+slot]={origin_id:3,origin_slot:0,target_id:1,target_slot:slot};}
  p.state.media=[{kind:'image',slot:0,path:'original.png'}];
  const result={id:'new',path:'generated.png',name:'generated.png',preview_url:'/h3fs/file/new'};p.generatedImage(source,result);
  assert.equal(p.state.media.find(x=>x.slot===0).path,'original.png');assert.equal(p.state.media.find(x=>x.slot===1).path,'generated.png');assert.equal(p.state.media.find(x=>x.slot===2).path,'generated.png');
  assert.equal(p.mediaList.querySelectorAll('button').filter(x=>x.textContent==='应用此图片').length,1);assert.equal(p.mediaList.querySelectorAll('img').length,3);
  p.mediaList.querySelectorAll('button').find(x=>x.textContent==='应用此图片').onclick();assert.equal(p.state.media.find(x=>x.slot===0).path,'generated.png');assert.equal(p.node.properties.h3fs_pending_images[0],undefined);
});

test('loading old embedded generation settings archives them without deleting material paths',()=>{
  const p=renderFixture();const old={auto_reference:true,reference_one_click:true,reference_cards:[{slot:0,text:'美女'}],media:[{kind:'image',slot:0,path:'kept.png'}]};p.node.widgets.find(w=>w.name==='settings').value=JSON.stringify(old);p.readState();
  assert.equal(p.state.auto_reference,false);assert.equal(p.state.reference_one_click,false);assert.equal(p.state.media[0].path,'kept.png');assert.equal(p.node.properties.h3fs_legacy_reference_settings.reference_cards[0].text,'美女');
});

test('compact workbench keeps common fields visible and moves low-frequency controls to settings',()=>{
  const p=renderFixture();assert.ok(p.tabPages.materials);assert.ok(p.tabPages.results.hidden);assert.ok(p.tabPages.settings.hidden);
  assert.ok(p.root.querySelectorAll('.h3-fast-row')[0].querySelectorAll('input').includes(p.controls.seconds));
  assert.ok(p.tabPages.settings.querySelectorAll('select').includes(p.controls.llm_model));
  p.selectTab('settings');assert.equal(p.tabPages.settings.hidden,false);assert.equal(p.tabPages.materials.hidden,true);
});
test('overall mode exposes nine independent checkboxes and intent fields without legacy editor complexity',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.render();assert.equal(p.mediaList.querySelectorAll('.h3-global-slot').length,9);
  const card=p.mediaList.querySelectorAll('.h3-global-slot')[1],check=card.querySelectorAll('input')[0],noun=card.querySelectorAll('input')[1];
  check.checked=true;check.onchange();noun.value='小院子';noun.oninput();assert.equal(p.globalCard(1).enabled,true);assert.equal(p.globalCard(1).text,'小院子');assert.equal(p.globalCard(0).enabled,false);
});
test('each image has compact aspect defaulting to video and overall cards have explicit independent type',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.render();
  const controls=p.mediaList.querySelectorAll('select'),find=name=>controls.find(x=>x.attributes['aria-label']===name);
  assert.equal(find('图片 1 画幅').value,'video');assert.equal(find('图片 2 素材类型').value,'auto');
  find('图片 2 素材类型').value='scene';find('图片 2 素材类型').onchange();
  find('图片 2 画幅').value='9:16';find('图片 2 画幅').onchange();
  assert.equal(p.globalCard(1).kind,'scene');assert.equal(p.globalCard(1).aspect,'9:16');assert.equal(p.globalCard(0).aspect,'video');
  p.state.global_reference.enabled=false;p.render();assert.equal(p.mediaList.querySelectorAll('.h3-slot-options').length,9);
});
test('portable aspect selector updates an attached generator rather than only changing its label',()=>{
  const p=renderFixture(),source={type:'H3ImageGenerator',h3ImagePanel:{state:{aspect:'1:1'},save(){this.saved=true;},render(){this.rendered=true;}}};
  p.generatorFor=()=>source;p.render();const select=p.mediaList.querySelectorAll('select').find(x=>x.attributes['aria-label']==='图片 1 画幅');
  select.value='9:16';select.onchange();assert.equal(source.h3ImagePanel.state.aspect,'9:16');assert.equal(source.h3ImagePanel.saved,true);
  select.value='video';select.onchange();assert.equal(source.h3ImagePanel.state.aspect,p.state.aspect);
});
test('batch material completion fills only selected slots and switches to video without discarding new intent',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.globalCard(0).enabled=true;p.state.media=[{kind:'image',slot:1,path:'manual.png'}];p.render();p.node.widgets[0].value='新的视频意图';
  p.globalResult({complete:true,job:{id:'a'.repeat(32),cards:{0:{slot:0,path:'generated.png'},1:{slot:1,path:'old.png'}},status:'完成'}});
  assert.equal(p.state.global_reference.action,'video');assert.equal(p.state.media.find(x=>x.slot===0).path,'generated.png');assert.equal(p.state.media.find(x=>x.slot===1).path,'manual.png');assert.equal(p.currentIntent(),'新的视频意图');
});
test('overall model family preserves separate LoRA combinations and sampling settings',()=>{
  const p=renderFixture();p.cfg.models.diffusion_models=['krea2_turbo.safetensors','qwen_image_2.1_bf16.safetensors'];p.openGlobalSettings=()=>{};
  p.state.global_reference.image_settings={image_engine:'krea2',image_model:'krea2_turbo.safetensors',image_steps:8,image_loras:[{name:'CML-Krea.safetensors',weight:1,enabled:true}]};
  p.globalFamily('qwen21');assert.equal(p.globalImageSettings().image_steps,25);p.globalImageSettings().image_loras=[{name:'CML-Qwen.safetensors',weight:.8,enabled:true}];
  p.globalFamily('krea2');assert.equal(p.globalImageSettings().image_loras[0].name,'CML-Krea.safetensors');p.globalFamily('qwen21');assert.equal(p.globalImageSettings().image_loras[0].weight,.8);
});
test('live MP4 is seekable and paused sampling is not overwritten by the next step',()=>{
  const {panel}=fixture();panel.previewRefresh=new Element('button');panel.previewEvent({image:'step1',mime:'video/mp4',step:1,total:4});const v=panel.preview.firstElementChild;assert.equal(v.controls,true);
  panel.previewReplacing=false;v.onpause();panel.previewEvent({image:'step2',mime:'video/mp4',step:2,total:4});assert.match(v.src,/step1$/);assert.equal(panel.pendingPreview.image,'step2');assert.equal(panel.previewRefresh.hidden,false);
});

test('restoring the playback position does not count as a user pause or seek',()=>{
  const {panel}=fixture();panel.previewEvent({image:'step1',mime:'video/mp4'});const v=panel.preview.firstElementChild;
  v.duration=10;v.currentTime=3;panel.previewEvent({image:'step2',mime:'video/mp4'});v.onloadedmetadata();v.onseeking();
  assert.notEqual(panel.previewPaused,true);v.onseeked();v.onseeking();assert.equal(panel.previewPaused,true);
});

test('restoring batch records cannot overwrite a later individual retake or a changed card intent',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.globalCard(0).enabled=true;p.globalCard(0).text='人物';p.globalCard(1).enabled=true;p.globalCard(1).text='新院子';
  p.state.media=[{kind:'image',slot:0,path:'individual-retake.png'}];
  p.globalResult({complete:true,job:{id:'a'.repeat(32),cards:{0:{slot:0,text:'人物',path:'old.png'},1:{slot:1,text:'旧院子',path:'wrong.png'}},status:'完成'}},true);
  assert.equal(p.state.media[0].path,'individual-retake.png');assert.equal(p.state.media.length,1);
});
test('a late batch result cannot overwrite a newly selected material type or aspect',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.globalCard(0).enabled=true;p.globalCard(0).text='庭院';p.globalCard(0).kind='scene';p.globalCard(0).aspect='9:16';
  p.globalResult({complete:true,job:{id:'a'.repeat(32),cards:{0:{slot:0,text:'庭院',kind:'person',aspect:'video',path:'stale.png'}},status:'完成'}});
  assert.equal(p.state.media.length,0);
});

test('video after prompt review reuses the current design but a changed intent requires reoptimization',()=>{
  const {panel,node}=fixture();panel.state.reviewed_prompt='已生成的中文H3提示词';panel.optimizedPrompt.value=panel.state.reviewed_prompt;node.properties.h3fs_prompt_origin='An idea';
  panel.prepareGlobalVideo();assert.equal(panel.state.use_reviewed_prompt,true);assert.equal(panel.state.optimize_on_run,false);assert.equal(panel.state.generate_after_optimization,true);
  node.widgets[0].value='New idea';panel.invalidateReviewed();panel.prepareGlobalVideo();assert.equal(panel.state.optimize_on_run,true);assert.equal(panel.state.use_reviewed_prompt,false);
});

test('normal video button cannot silently reuse a stale prompt after the intent changes',async()=>{
  const {panel,node}=fixture();panel.state.reviewed_prompt='旧提示词';panel.state.optimize_on_run=true;node.properties.h3fs_prompt_origin='旧意图';panel.render=()=>{};app.queuePrompt=async()=>{};
  await panel.queueVideo(false);assert.notEqual(panel.state.use_reviewed_prompt,true);assert.equal(panel.state.optimize_on_run,true);
});

test('failed candidate stays visible but cannot be approved until explicitly edited',()=>{
  const p=renderFixture();p.setPromptCandidate({text:'未完成候选',original:'An idea',stage:'error',warnings:['标签不完整']});
  assert.equal(p.optimizedPrompt.value,'未完成候选');assert.equal(p.reviewButton.disabled,true);assert.equal(p.state.use_reviewed_prompt,false);
  p.prepareGlobalVideo();assert.equal(p.state.optimize_on_run,true);
  p.openVideoPrompt();const input=p.dialog.querySelectorAll('textarea')[0];assert.equal(input.value,'未完成候选');input.value='人工修正后的完整提示词';
  p.dialog.querySelectorAll('button').find(x=>x.textContent==='保存修改').onclick();assert.equal(p.node.properties.h3fs_prompt_state.stage,'edited');assert.equal(p.reviewButton.disabled,false);
});

test('reloading workflow preserves failed-candidate review guard',()=>{
  const p=renderFixture();p.node.properties.h3fs_prompt_state={stage:'error',text:'候选'};p.readState();p.optimizedPrompt.value='候选';p.updateReviewButton();assert.equal(p.reviewButton.disabled,true);
});

test('each batch picture exposes its actual prompt, editable without making a longer main panel',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.node.properties.h3fs_global_job={cards:{1:{slot:1,prompt:'独立庭院实景，柔光，无人物。'}}};p.render();
  p.mediaList.querySelectorAll('button').find(x=>x.attributes['aria-label']==='图片 2 查看 / 编辑提示词').onclick();
  const input=p.dialog.querySelectorAll('textarea')[0];assert.equal(input.value,'独立庭院实景，柔光，无人物。');input.value='庭院，黄昏，无人物。';
  p.dialog.querySelectorAll('button').find(x=>x.textContent==='保存本图提示词').onclick();assert.equal(p.globalCard(1).draft,input.value);
  assert.equal(p.mediaList.querySelectorAll('textarea').length,0);
});

test('single batch retake queues only that frame and never the downstream video outputs',async()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.state.global_reference.job_id='a'.repeat(32);for(const i of [0,1])Object.assign(p.globalCard(i),{enabled:true,text:'庭院',draft:'庭院实景。'});
  app.graphToPrompt=async()=>({workflow:{id:'workflow-a'},output:{1:{class_type:'H3FullStack',inputs:{prompt:'An idea',settings:JSON.stringify(p.state)}},7:{class_type:'SaveVideo',inputs:{video:['1',0]}}}});
  let sent;api.queuePrompt=async(_,data)=>sent=data;await p.queueGlobal('images',1);assert.ok(sent,p.statusEl.textContent);
  assert.deepEqual(Object.keys(sent.output),['1']);const c=JSON.parse(sent.output[1].inputs.settings);assert.equal(c.global_reference.action,'images');assert.equal(c.global_reference.cards.filter(x=>x.enabled).length,1);assert.equal(c.global_reference.cards.find(x=>x.enabled).slot,1);
  assert.equal(p.globalCard(0).enabled,true);
});

test('an execution error retains both prompt and images and clearly stops progress',()=>{
  const p=renderFixture();panels.clear();panels.set(p.node,p);app.graph=p.node.graph;p.setPromptCandidate({text:'完整提示词',original:'An idea',stage:'ready'});p.state.media=[{kind:'image',slot:0,path:'kept.png'}];
  listeners.execution_error({detail:{node_id:'1.4',exception_message:'测试采样失败'}});
  assert.equal(p.optimizedPrompt.value,'完整提示词');assert.equal(p.state.media[0].path,'kept.png');assert.match(p.statusEl.textContent,/已保留/);assert.equal(p.globalStage,'error');
});

test('saved video prompt restoration checks duration, language and actual assets',async()=>{
  const p=renderFixture();p.state.mode='reference';p.state.seconds=10;p.state.global_reference.enabled=true;p.state.global_reference.job_id='b'.repeat(32);p.state.media=[{kind:'image',slot:0,path:'same.png'}];
  const checkpoint={text:'保存的完整提示词',stage:'ready',original:'An idea',mode:'reference',seconds:10,use_chinese_prompt:true,media_paths:['same.png']};
  api.fetchApi=async()=>({ok:true,json:async()=>({id:'b'.repeat(32),cards:{},video_prompt:checkpoint})});
  await p.restoreGlobal();assert.equal(p.optimizedPrompt.value,checkpoint.text);
  p.optimizedPrompt.value='保留当前文字';p.state.seconds=5;p.restoredGlobalId=null;await p.restoreGlobal();assert.equal(p.optimizedPrompt.value,'保留当前文字');
  p.state.seconds=10;p.state.use_chinese_prompt=false;p.restoredGlobalId=null;await p.restoreGlobal();assert.equal(p.optimizedPrompt.value,'保留当前文字');
  p.state.use_chinese_prompt=true;p.state.media[0].path='new.png';p.restoredGlobalId=null;await p.restoreGlobal();assert.equal(p.optimizedPrompt.value,'保留当前文字');
});

test('video handoff updates the batch progress instead of remaining on completed images',()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;panels.clear();panels.set(p.node,p);app.graph=p.node.graph;
  listeners.h3fs_status({detail:{node_id:1,text:'reference · 416×256 · 56 frames / 2.333s'}});assert.equal(p.globalStage,'video');
  listeners.progress({detail:{node:1,value:2,max:4}});assert.equal(p.globalProgressBar.value,2);assert.match(p.globalProgressLabel.textContent,/2 \/ 4/);
  listeners.h3fs_result({detail:{node_id:1,files:[{url:'/base.mp4'}]}});assert.equal(p.globalStage,'complete_video');assert.match(p.statusEl.textContent,/完整视频已保存/);
});

test('overall video run includes connected exports and their other ancestors, not unrelated nodes',async()=>{
  const p=renderFixture();p.state.global_reference.enabled=true;p.state.global_reference.job_id='c'.repeat(32);Object.assign(p.globalCard(0),{enabled:true,text:'院子'});
  app.graphToPrompt=async()=>({workflow:{id:'workflow-a'},output:{1:{class_type:'H3FullStack',inputs:{prompt:'An idea',settings:JSON.stringify(p.state)}},7:{class_type:'SaveVideo',inputs:{video:['1',0]}},8:{class_type:'SaveVideo',inputs:{video:['1',1]}},9:{class_type:'ShowText|pysssss',inputs:{text:['1',2]}},10:{class_type:'SaveImage',inputs:{images:['11',0]}},11:{class_type:'EmptyImage',inputs:{width:64,height:64}}}});
  let sent;api.queuePrompt=async(_,data)=>sent=data;await p.queueGlobal('video');assert.deepEqual(Object.keys(sent.output).sort(),['1','7','8','9']);
  await p.queueGlobal('prompt');assert.deepEqual(Object.keys(sent.output),['1']);
});
test('postprocess updates progress and base result remains previewable before upscaling ends',()=>{
  const p=renderFixture();panels.clear();panels.set(p.node,p);app.graph=p.node.graph;
  listeners.h3fs_partial_result({detail:{node_id:1,files:[{url:'/saved-base.mp4'}],text:'原视频已保存'}});assert.match(p.preview.firstElementChild.src,/saved-base.mp4$/);
  listeners.h3fs_postprocess({detail:{node_id:1,stage:'encode',text:'编码 24 / 48 帧',done:24,total:48}});assert.equal(p.globalProgressBar.value,24);assert.equal(p.globalStage,'encode');assert.match(p.statusEl.textContent,/编码/);
});

test('restoring a prompt checkpoint cannot suppress video history; late history cannot replace new playback',async()=>{
  vm.runInContext('historyRequest=null',context);
  const p=renderFixture(),history=historyFixture();let release;const result=new Promise(resolve=>release=resolve);
  history.success.outputs={1:{h3fs_files:[{url:'/latest.mp4'}],h3fs_final_prompt:['完整提示词']}};
  api.fetchApi=async()=>({ok:true,json:async()=>result});
  const first=p.restoreHistory();p.setPromptCandidate({text:'完整提示词',original:'An idea',stage:'ready'});release(history);await first;
  assert.match(p.preview.firstElementChild.src,/latest.mp4$/);
  const second=p.restoreHistory();p.renderResults([{url:'/new.mp4'}]);await second;assert.match(p.preview.firstElementChild.src,/new.mp4$/);
  const third=p.restoreHistory();p.startPreview();await third;assert.equal(p.preview.children.length,0);
  vm.runInContext('historyRequest=null',context);
});

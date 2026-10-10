import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { openAPISettings } from './api_settings.js';

const TYPE = 'H3ImageGenerator';
const instances = new Map();
const modes = [['style_intent','参考画风＋输入意向'],['recreate','完全复刻'],['character_card','人物卡片']];
const el = (tag, cls='', text='') => { const n=document.createElement(tag); n.className=cls; n.textContent=text; return n; };
const widget = (n,k) => n.widgets?.find(w=>w.name===k);
function hideNativeWidgets(node) {
  for(const key of ['settings','intent','seed','control_after_generate']) {
    const w=widget(node,key);if(!w)continue;
    w.type='h3fs_hidden';w.computeSize=()=>[0,-4];w.draw=()=>{};w.mouse=()=>false;
    if(w.inputEl)w.inputEl.style.display='none';
    // Re-rolling is explicit. Running a downstream video must keep the image seed.
    if(key==='control_after_generate')w.value='fixed';
  }
}
const btn = (text,fn) => { const b=el('button','',text); b.type='button'; b.onclick=fn; return b; };
const imageUrl = result => result.preview_url ? api.apiURL(result.preview_url) : api.apiURL('/view?'+new URLSearchParams(result.image));
const upstream = (graph, links) => {const out={}; const visit=link=>{if(!Array.isArray(link)||link.length!==2)return;const id=String(link[0]);if(!graph[id]||out[id])return;out[id]=graph[id];Object.values(graph[id].inputs).forEach(visit);};links.forEach(visit);return out;};

class ImagePanel {
  constructor(node,cfg) {
    this.node=node; this.cfg=cfg; this.root=el('div','h3fs h3-image-generator');
    for(const event of ['pointerdown','wheel','keydown'])this.root.addEventListener(event,e=>e.stopPropagation());
    this.read(); this.render(); this.restore();
  }
  read() {
    let saved={}; try{saved=JSON.parse(widget(this.node,'settings').value||'{}');}catch{}
    this.state={...structuredClone(this.cfg.generator_defaults),...saved};
    if(this.cfg.generator_legacy_card_template&&this.state.card_template?.trim()===this.cfg.generator_legacy_card_template)this.state.card_template=this.cfg.generator_defaults.card_template;
    this.state.instance_id ||= crypto.randomUUID();
    this.state.owner_id=String(this.node.id);
  }
  save() {widget(this.node,'settings').value=JSON.stringify(this.state);this.node.graph?.change();this.node.graph?.setDirtyCanvas(true,true);}
  clearDraft() {this.state.draft='';this.state.draft_key='';this.state.draft_manual=false;if(this.prompt)this.prompt.value='';}
  field(parent,key,label,{values,type='text',min,max,step,change}={}) {
    const l=el('label','',label),input=el(values?'select':'input');input.setAttribute('aria-label',label);
    if(values)for(const entry of values){const [value,text]=Array.isArray(entry)?entry:[entry,entry];const option=el('option','',text);option.value=value;input.append(option);}
    else {input.type=type;if(min!=null)input.min=min;if(max!=null)input.max=max;if(step!=null)input.step=step;}
    // Do not silently discard a saved custom model/encoder when it is unavailable.
    if(values&&!values.some(v=>(Array.isArray(v)?v[0]:v)===this.state[key])&&this.state[key]){const o=el('option','',this.state[key]+'（当前未安装）');o.value=this.state[key];input.prepend(o);}
    input.value=this.state[key]??'';
    input.onchange=()=>{if(!input.checkValidity())return;this.state[key]=type==='number'?Number(input.value):input.value;this.save();change?.();};
    l.append(input);parent.append(l);return input;
  }
  fold(parent,title,open=false) {const d=el('details'),s=el('summary','',title);d.open=open;d.append(s);parent.append(d);return d;}
  text(parent,label,value,oninput) {const l=el('label','',label),t=el('textarea');t.value=value||'';t.setAttribute('aria-label',label);t.oninput=()=>{oninput(t.value);this.save();};l.append(t);parent.append(l);return t;}
  familyFiles(category) {const qwen=this.state.image_engine==='qwen21';return this.cfg.models[category].filter(n=>category==='diffusion_models'?qwen?/qwen_image_2[._]1/i.test(n):/krea2/i.test(n):category==='loras'?qwen?/qwen|qwen2\.1/i.test(n):/krea|kera/i.test(n):true);}
  familyChanged() {
    const qwen=this.state.image_engine==='qwen21',files=this.familyFiles('diffusion_models');
    this.state.image_model=files.find(n=>qwen?/bf16/i.test(n):/turbo/i.test(n))||files[0]||'';
    this.state.image_clip=this.cfg.models.text_encoders.find(n=>qwen?/qwen3vl_8b_bf16/i.test(n):/qwen3vl_4b/i.test(n))||'';
    this.state.image_vae=this.cfg.models.vae.find(n=>qwen?/qwen_image_2\.1/i.test(n):/qwen_image_vae\.safetensors$/i.test(n))||'';
    this.state.image_steps=qwen?25:8;
    // Explicitly disable incompatible family selections instead of silently attaching them.
    for(const row of this.state.image_loras)if(!this.familyFiles('loras').includes(row.name))row.enabled=false;
    this.save();this.render();
  }
  render() {
    const scroll=this.root.scrollTop;this.root.replaceChildren();
    this.root.append(el('div','h3-generator-title','参考图生图'),el('p','h3-help','独立生图 · 可连接多个图片框 · 不自动启动视频'));
    this.root.append(btn('提示词模型 · 本地 / API',()=>openAPISettings(this.state,this.cfg,()=>this.save())));
    const model=el('div','h3-row2');this.root.append(model);
    this.field(model,'image_engine','生图模型',{values:[['krea2','Krea2'],['qwen21','Qwen Image 2.1（文生图）']],change:()=>this.familyChanged()});
    this.field(model,'aspect','图片比例',{values:['16:9','9:16','1:1','4:3','3:4','3:2']});
    const choices=el('div','h3-row2');this.root.append(choices);
    this.field(choices,'image_model','模型文件',{values:this.familyFiles('diffusion_models')});
    this.field(choices,'constraint','参考约束',{values:modes,change:()=>{this.state.draft='';this.state.draft_key='';if(this.state.constraint==='character_card')this.state.aspect='3:2';this.save();this.render();}});
    const explanation={recreate:'反推主体、姿势、服装、构图、背景和光线，再文生图；不是图片编辑，不能保证像素级一比一。',character_card:'生成一张正／侧／背三视图卡片。模板可修改；默认不反推五官，脸由 LoRA 控制。',style_intent:'只学习参考图的光线、色调和画风；画什么由下面的输入决定。不加参考图时直接使用原文生图。'};
    this.root.append(el('p','h3-help',explanation[this.state.constraint]));
    const upload=el('div','h3-generator-reference');upload.classList.toggle('has-reference',!!this.state.reference_path);this.root.append(upload);
    const input=el('input');input.type='file';input.accept='image/png,image/jpeg,image/webp';input.hidden=true;
    input.onchange=()=>this.upload(input.files[0]);upload.append(input);
    const referenceButton=btn(this.state.reference_path?'替换参考图':'参考图（可选）：拖入或点击上传',()=>input.click());upload.append(referenceButton);
    if(this.state.reference_path){const thumb=el('img');thumb.src=api.apiURL('/view?'+new URLSearchParams({filename:this.state.reference_path.split('/').pop(),subfolder:'h3_fullstack',type:'input'}));thumb.alt='参考图';thumb.onclick=()=>this.view(thumb.src);upload.append(thumb,btn('移除参考图',()=>{this.state.reference_path='';this.state.reference_name='';this.state.draft='';this.save();this.render();}));}
    upload.ondragover=e=>{e.preventDefault();e.stopPropagation();};upload.ondrop=e=>{e.preventDefault();e.stopPropagation();this.upload(e.dataTransfer.files[0]);};
    this.intent=this.text(this.root,'输入意向／提示词',widget(this.node,'intent').value,v=>{widget(this.node,'intent').value=v;this.clearDraft();});
    this.intent.placeholder=this.state.constraint==='recreate'?'可以留空，按参考图反推':this.state.constraint==='character_card'?'可填写服装补充、人物触发词 CML / XNJ':'例如：小院子，清晨，低机位；或粘贴完整提示词';
    const loras=this.fold(this.root,'LoRA 与权重',this.state.image_loras.length>0);this.loraList=el('div');loras.append(this.loraList);
    this.renderLoras();loras.append(btn('＋ 添加 LoRA',()=>{this.state.image_loras.push({name:this.familyFiles('loras').find(n=>!/turbo_sda/i.test(n))||'',weight:1,enabled:true});this.save();this.renderLoras();loras.open=true;}));
    const run=el('div','h3-generator-run');this.root.append(run);this.generateButton=btn('生成图片',()=>this.queue('generate'));run.append(this.generateButton,btn('仅生成提示词',()=>this.queue('design')));
    this.progress=el('progress');this.progress.max=2;this.progress.hidden=true;this.status=el('div','h3-status');this.status.setAttribute('role','status');this.root.append(this.progress,this.status);
    const prompt=this.fold(this.root,'本次生图提示词（可直接修改）',!!this.state.draft);this.prompt=this.text(prompt,'生图提示词',this.state.draft,v=>{this.state.draft=v;this.state.draft_manual=true;});
    if(this.state.constraint==='character_card'){const template=this.fold(this.root,'人物卡片内置模板（可编辑）');this.text(template,'人物卡片模板',this.state.card_template,v=>{this.state.card_template=v;this.clearDraft();});}
    const advanced=this.fold(this.root,'更多参数');const quality=el('div','h3-row2');advanced.append(quality);
    this.field(quality,'image_steps','采样步数',{type:'number',min:1,max:100,step:1});this.field(quality,'image_megapixels','清晰度（百万像素）',{type:'number',min:.1,max:4,step:.1});
    const seed=el('label','','种子（相同参数可复现）'),number=el('input');number.type='number';number.min=0;number.max=Number.MAX_SAFE_INTEGER;number.step=1;number.value=widget(this.node,'seed').value;number.setAttribute('aria-label','种子');number.onchange=()=>{if(number.checkValidity()){widget(this.node,'seed').value=Number(number.value);this.save();}};seed.append(number);advanced.append(seed);
    this.field(advanced,'image_cfg','CFG',{type:'number',min:0,max:20,step:.1});
    for(const[k,label,cat]of [['image_clip','图片文本编码器','text_encoders'],['image_vae','图片 VAE','vae'],['llm_model','反推用本地模型','LLM'],['llm_mmproj','视觉投影模型','LLM']])this.field(advanced,k,label,{values:this.cfg.models[cat].filter(n=>cat!=='LLM'||(k==='llm_model'?!/mmproj/i.test(n):/mmproj/i.test(n)))});
    this.field(advanced,'llm_handler','视觉模型模板',{values:['Qwen3.8','Qwen3.5','Qwen3-VL']});
    this.resultArea=el('div','h3-generator-result');this.root.append(this.resultArea);if(this.node.properties.h3fs_image_result)this.showResult(this.node.properties.h3fs_image_result,false);
    this.save();this.root.scrollTop=scroll;
  }
  renderLoras() {
    this.loraList.replaceChildren();for(const[index,row]of this.state.image_loras.entries()){
      const line=el('div','h3-lora'),enabled=el('input');enabled.type='checkbox';enabled.checked=row.enabled;enabled.setAttribute('aria-label',`LoRA ${index+1} 启用`);enabled.onchange=()=>{row.enabled=enabled.checked;this.save();};
      const name=el('select');name.setAttribute('aria-label',`LoRA ${index+1} 模型`);const files=this.familyFiles('loras');if(row.name&&!files.includes(row.name))files.unshift(row.name);for(const file of files){const option=el('option','',file);option.value=file;name.append(option);}name.value=row.name;name.onchange=()=>{row.name=name.value;this.save();};
      const weight=el('input');weight.type='number';weight.min=-5;weight.max=5;weight.step=.05;weight.value=row.weight;weight.setAttribute('aria-label',`LoRA ${index+1} 权重`);weight.onchange=()=>{if(weight.checkValidity()){row.weight=Number(weight.value);this.save();}};
      line.append(enabled,name,weight,btn('×',()=>{this.state.image_loras.splice(index,1);this.save();this.renderLoras();}));this.loraList.append(line);
    }
  }
  async upload(file) {if(!file)return;try{const body=new FormData();body.append('file',file);const r=await api.fetchApi('/h3fs/upload',{method:'POST',body}),data=await r.json();if(!r.ok||data.kind!=='image')throw Error(data.error||'请上传图片');this.state.reference_path=data.path;this.state.reference_name=data.name;this.state.draft='';this.state.draft_key='';this.save();this.render();}catch(e){this.setStatus(e.message,true);}}
  setStatus(text,error=false) {this.status.textContent=text;this.status.classList.toggle('h3-error',error);}
  async queue(action, reusePrompt=false) {
    if(this.busy)return;this.busy=true;this.generateButton.disabled=true;
    try{if(reusePrompt&&this.state.draft.trim())this.state.draft_manual=true;this.state.action=action;this.state.nonce=crypto.randomUUID();this.state.owner_id=String(this.node.id);this.state.force_regenerate=false;this.save();const graph=await app.graphToPrompt();const inputs=graph.output[String(this.node.id)]?.inputs;if(!inputs)throw Error('节点不在当前画布中');const output=upstream(graph.output,[inputs.reference]);output[String(this.node.id)]={class_type:TYPE,inputs:{...inputs,settings:JSON.stringify({...this.state,force_regenerate:action==='generate'})}};await api.queuePrompt(0,{output,workflow:graph.workflow});this.setStatus(reusePrompt?'复用当前提示词，直接生图；不会启动视频':'图片任务已加入队列；不会运行下游视频');}catch(e){this.setStatus(e.message,true);}finally{this.busy=false;this.generateButton.disabled=false;}
  }
  promptResult(data) {this.state.draft=data.text;this.state.draft_key=data.draft_key;this.state.draft_manual=false;this.prompt.value=data.text;this.prompt.closest('details').open=true;this.save();}
  showResult(result,broadcast=true) {
    if(broadcast&&this.prompt)this.prompt.closest('details').open=false;
    this.node.properties.h3fs_image_result=result;this.resultArea.replaceChildren();const image=el('img');image.src=imageUrl(result);image.alt='生成的参考图';image.onclick=()=>this.view(image.src);this.resultArea.append(image);
    const actions=el('div','h3-generator-run');actions.append(btn('再抽一张',()=>{widget(this.node,'seed').value=(Number(widget(this.node,'seed').value)+1)%Number.MAX_SAFE_INTEGER;this.save();this.render();this.queue('generate',true);}),btn('应用到已连接图片框',()=>{window.dispatchEvent(new CustomEvent('h3fs-apply-generator',{detail:{source:this.node,result}}));}));this.resultArea.append(actions);
    this.state.result_id=result.id;this.save();if(broadcast)window.dispatchEvent(new CustomEvent('h3fs-generator-result',{detail:{source:this.node,result}}));
  }
  async restore() {if(!this.state.result_id)return;try{const r=await api.fetchApi('/h3fs/image-result/'+this.state.result_id);if(r.ok){this.showResult(await r.json());this.setStatus('已载入上次结果；可应用或再抽一张');}}catch(e){this.setStatus(e.message,true);}}
  view(url) {const d=el('dialog','h3fs-modal h3-image-viewer'),head=el('div','h3-dialog-head'),image=el('img','h3-full-image');image.src=url;head.append(el('h2','','原图预览'),btn('关闭',()=>d.close()));d.append(head,image);d.onclose=()=>d.remove();d.onclick=e=>{if(e.target===d)d.close();};document.body.append(d);d.showModal();}
}

app.registerExtension({name:'EverythingIsParty.H3ImageGenerator',async beforeRegisterNodeDef(Node,data){
  if(data.name!==TYPE)return;
  const created=Node.prototype.onNodeCreated;Node.prototype.onNodeCreated=function(){const r=created?.apply(this,arguments);this.properties??={};
    // The DOM panel owns its one result preview. Keep native history/Assets output,
    // but do not also add Comfy's default canvas-image widget below the panel.
    this.onDrawBackground=()=>{};
    hideNativeWidgets(this);
    this.widgets_start_y=50;const holder=el('div');holder.style.cssText='width:100%;height:100%;min-width:390px';const dom=this.addDOMWidget('reference_image_generator','h3fs_image_generator',holder,{serialize:false,hideOnZoom:false});dom.computeSize=()=>[410,680];dom.options.getMinHeight=()=>680;dom.options.getMaxHeight=()=>680;this.size=[450,780];this.color='#253f43';this.bgcolor='#202629';
    api.fetchApi('/h3fs/config').then(r=>r.json()).then(cfg=>{if(this._h3Removed)return;const panel=new ImagePanel(this,cfg);holder.replaceChildren(panel.root);this.h3ImagePanel=panel;instances.set(this,panel);this.setSize([450,780]);}).catch(e=>holder.textContent=e.message);return r;};
  const configured=Node.prototype.onConfigure;Node.prototype.onConfigure=function(){const r=configured?.apply(this,arguments);hideNativeWidgets(this);if(this.h3ImagePanel){this.h3ImagePanel.read();this.h3ImagePanel.render();this.h3ImagePanel.restore();}return r;};
  const removed=Node.prototype.onRemoved;Node.prototype.onRemoved=function(){this._h3Removed=true;instances.delete(this);return removed?.apply(this,arguments);};
}});

function forEvent(detail){return [...instances.values()].find(p=>p.node.graph===app.graph&&String(p.node.id)===String(detail.node_id));}
api.addEventListener('h3fs_generator_prompt',e=>forEvent(e.detail)?.promptResult(e.detail));
api.addEventListener('h3fs_generator_result',e=>{const p=forEvent(e.detail),instance=e.detail.result.source_instance_id??e.detail.result.settings.instance_id;if(p&&(!instance||instance===p.state.instance_id))p.showResult(e.detail.result);});
api.addEventListener('h3fs_generator_progress',e=>{const p=forEvent(e.detail);if(!p)return;p.progress.hidden=false;p.progress.value=e.detail.done;p.setStatus(e.detail.text,e.detail.error);});

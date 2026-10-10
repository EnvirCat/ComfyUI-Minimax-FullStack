const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {test}=require('node:test');
class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.style={};this.classList={toggle(){}};this.scrollTop=0;}
  append(...children){for(const c of children)c.parent=this;this.children.push(...children);}
  replaceChildren(...children){this.children=[];this.append(...children);}
  prepend(...children){this.children.unshift(...children);}
  setAttribute(k,v){this.attributes[k]=v;}
  addEventListener(){}checkValidity(){return true;}
  closest(tag){let p=this;while(p&&p.tagName!==tag.toUpperCase())p=p.parent;return p;}
  query(tag){return this.children.flatMap(c=>[...(c.tagName===tag.toUpperCase()?[c]:[]),...c.query(tag)]);}
}
const events={};let sent;
const app={registerExtension(e){this.extension=e;},graphToPrompt:async()=>({workflow:{id:'test'},output:{3:{class_type:'H3ImageGenerator',inputs:{intent:'蛋糕',settings:'{}',seed:42,reference:['10',0]}},10:{class_type:'LoadImage',inputs:{image:'ref.png'}},1:{class_type:'H3FullStack',inputs:{image_1:['3',0]}},7:{class_type:'SaveVideo',inputs:{video:['1',0]}}}})};
const api={apiURL:x=>x,addEventListener(n,f){events[n]=f;},queuePrompt:async(_,data)=>sent=data};
const context=vm.createContext({app,api,console,URLSearchParams,structuredClone,crypto:{randomUUID:()=> 'test-uuid'},CustomEvent:class{constructor(type,value){this.type=type;this.detail=value.detail;}},window:{dispatchEvent(e){events[e.type]?.(e);}},document:{createElement:t=>new Element(t)}});
const src=fs.readFileSync(require('node:path').join(__dirname,'../web/image_generator.js'),'utf8').replace(/^import .*;\r?\n/gm,'');
vm.runInContext(src+'\nthis.subject={ImagePanel,upstream,hideNativeWidgets};',context);
function fixture(){const node={id:3,properties:{},widgets:[{name:'settings',value:'{}'},{name:'intent',value:'蛋糕'},{name:'seed',value:42}],graph:{change(){},setDirtyCanvas(){}}};const defaults={image_engine:'krea2',image_model:'krea2_turbo.safetensors',image_clip:'qwen3vl_4b.safetensors',image_vae:'qwen_image_vae.safetensors',image_loras:[],constraint:'style_intent',aspect:'9:16',reference_path:'',card_template:'正视图、侧视图、背视图',draft:''};const cfg={generator_defaults:defaults,models:{diffusion_models:['krea2_turbo.safetensors','qwen_image_2.1_bf16.safetensors'],text_encoders:['qwen3vl_4b.safetensors','qwen3vl_8b_bf16.safetensors'],vae:['qwen_image_vae.safetensors','qwen_image_2.1_vae.safetensors'],loras:['Krea2/CML.safetensors','QWEN2.1/CML.safetensors'],LLM:[]}};return new context.subject.ImagePanel(node,cfg);}
test('one simple node offers exactly three constraints and visible model/aspect choices',()=>{const p=fixture();const fields=p.root.query('select');const modes=fields.find(x=>x.attributes['aria-label']==='参考约束');assert.equal(modes.children.length,3);assert.equal(p.intent.attributes['aria-label'],'输入意向／提示词');assert.ok(fields.some(x=>x.attributes['aria-label']==='图片比例'));assert.ok(fields.some(x=>x.attributes['aria-label']==='模型文件'));});
test('model family switches matching encoder/VAE and visibly disables incompatible LoRAs',()=>{const p=fixture();p.state.image_loras=[{name:'Krea2/CML.safetensors',weight:1,enabled:true}];p.state.image_engine='qwen21';p.familyChanged();assert.match(p.state.image_clip,/8b/);assert.match(p.state.image_vae,/2.1/);assert.equal(p.state.image_steps,25);assert.equal(p.state.image_loras[0].enabled,false);});
test('LoRA name, weight and enabled occupy one row and remain independently editable',()=>{const p=fixture();p.state.image_loras=[{name:'Krea2/CML.safetensors',weight:1,enabled:true}];p.renderLoras();const row=p.loraList.children[0];assert.equal(row.children.length,4);const weight=row.query('input').find(x=>x.type==='number');weight.value=.65;weight.onchange();assert.equal(p.state.image_loras[0].weight,.65);});
test('generator-only queue traverses reference ancestors but never video descendants',async()=>{const p=fixture();await p.queue('generate');assert.deepEqual(Object.keys(sent.output).sort(),['10','3']);assert.equal(JSON.parse(sent.output['3'].inputs.settings).action,'generate');assert.equal(sent.output['3'].inputs.seed,42);});
test('editable reviewed prompt does not require a second LLM analysis',()=>{const p=fixture();p.promptResult({text:'奶油色褶边上衣',draft_key:'key'});assert.equal(p.prompt.closest('details').open,true);p.prompt.value='人工修改';p.prompt.oninput();assert.equal(p.state.draft,'人工修改');assert.equal(p.state.draft_manual,true);assert.equal(p.state.draft_key,'key');});
test('character card template is visibly editable and produces a single result area',()=>{const p=fixture();p.state.constraint='character_card';p.render();const template=p.root.query('textarea').find(x=>x.attributes['aria-label']==='人物卡片模板');template.value='三视图的新模板';template.oninput();assert.equal(p.state.card_template,'三视图的新模板');assert.equal(p.root.children.filter(x=>x.className==='h3-generator-result').length,1);});
test('native widgets cannot overlap the panel or randomize a reused image seed',()=>{
  const node={widgets:['settings','intent','seed','control_after_generate'].map(name=>({name,value:'randomize',inputEl:{style:{}}}))};
  context.subject.hideNativeWidgets(node);
  for(const w of node.widgets){assert.equal(w.type,'h3fs_hidden');assert.equal(w.inputEl.style.display,'none');assert.equal(w.mouse(),false);assert.deepEqual([...w.computeSize()],[0,-4]);}
  assert.equal(node.widgets[3].value,'fixed');
});
test('editing an intent or card template clears both the stale prompt and approval',()=>{
  const p=fixture();p.promptResult({text:'旧草稿',draft_key:'old'});p.state.draft_manual=true;p.intent.value='小院子';p.intent.oninput();
  assert.equal(p.state.draft,'');assert.equal(p.prompt.value,'');assert.equal(p.state.draft_key,'');assert.equal(p.state.draft_manual,false);
});
test('reroll with changed LoRA uses the current prompt verbatim and queues only diffusion',async()=>{
  const p=fixture();p.state.draft='已经设计好的中文提示词';p.state.draft_key='previous-key';p.state.image_loras=[{name:'新LoRA.safetensors',weight:.8,enabled:true}];
  await p.queue('generate',true);const settings=JSON.parse(sent.output['3'].inputs.settings);assert.equal(settings.draft_manual,true);assert.equal(settings.draft,'已经设计好的中文提示词');assert.equal(settings.force_regenerate,true);
});

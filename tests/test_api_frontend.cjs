const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');const path=require('node:path');const {test}=require('node:test');
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.listeners={};this.classList={remove(){},add(){}};this.attributes={};}
  append(...x){this.children.push(...x);}setAttribute(k,v){this.attributes[k]=v;}addEventListener(e,fn){this.listeners[e]=fn;}
  showModal(){this.open=true;}close(){this.open=false;this.listeners.close?.();}remove(){this.removed=true;}
  query(tag){return this.children.flatMap(x=>[...(x.tagName===tag?[x]:[]),...x.query(tag)]);}
}
function fixture(language='zh') {
  const calls=[],state={ui_language:language,llm_backend:'local'},saved=[];
  const context=vm.createContext({document:{createElement:t=>new Element(t),body:new Element('body')},api:{fetchApi:async(url,o)=>{const body=JSON.parse(o.body);calls.push({url,body});return {ok:true,json:async()=>({has_key:body.action==='save',storage:'windows-dpapi',ok:true})};}}});
  const source=fs.readFileSync(path.join(__dirname,'../web/api_settings.js'),'utf8').replace(/^import .*;\r?\n/gm,'').replace('export function','function');
  vm.runInContext(source+'\nthis.openAPISettings=openAPISettings;',context);
  const cfg={api_presets:{openai:{label:'OpenAI',base_url:'https://api.openai.com/v1'},custom:{label:'Custom',base_url:''}}};
  const dialog=context.openAPISettings(state,cfg,()=>saved.push(JSON.stringify(state)));
  const select=dialog.query('select');select[1].value='openai';
  const field=name=>dialog.query('input').find(n=>n.attributes['aria-label']===name);
  const button=text=>dialog.query('button').find(n=>n.textContent===text);
  return {dialog,calls,state,saved,select,field,button};
}
test('API connection settings start local, keep advanced collapsed, and disclose upload',()=>{
  const f=fixture();assert.equal(f.select[0].value,'local');assert.ok(f.dialog.query('label').some(n=>n.textContent.includes('会上传参考图')));assert.notEqual(f.dialog.query('details')[0].open,true);
});
test('saving a key sends it only to local credential route, never workflow state',async()=>{
  const f=fixture();f.select[0].value='api';f.select[0].onchange();f.field('模型名称（按账号权限填写）').value='example-model';f.field('API 密钥（留空保留已存密钥）').value='test-only-private';
  await f.button('保存设置').onclick();assert.equal(f.state.llm_backend,'api');assert.equal(f.state.llm_api.model,'example-model');assert.ok(f.calls.some(c=>c.body.key==='test-only-private'));assert.ok(f.saved.every(s=>!s.includes('test-only-private')));assert.equal(f.field('API 密钥（留空保留已存密钥）').value,'');
});
test('text test never sends an image or enqueues video',async()=>{
  const f=fixture();f.select[0].value='api';f.field('模型名称（按账号权限填写）').value='test';await f.button('保存并测试文本连接').onclick();
  const testCall=f.calls.find(c=>c.url==='/h3fs/api/test');assert.ok(testCall);assert.deepEqual(Object.keys(testCall.body),['settings']);assert.ok(f.calls.every(c=>c.url.startsWith('/h3fs/api/')));
});
test('changing provider clears typed password and replaces endpoint, not saved key content',()=>{
  const f=fixture();const key=f.field('API 密钥（留空保留已存密钥）');key.value='typed-secret';f.select[1].value='custom';f.select[1].onchange();assert.equal(key.value,'');assert.equal(f.field('接口地址（Base URL）').value,'');
});
test('closing dialog wipes typed password and removes modal',()=>{
  const f=fixture();const key=f.field('API 密钥（留空保留已存密钥）');key.value='private';f.dialog.close();assert.equal(key.value,'');assert.ok(f.dialog.removed);
});
test('English UI has the same API and privacy controls',()=>{const f=fixture('en');assert.ok(f.field('API key (blank keeps saved key)'));assert.ok(f.button('Save & test text connection'));});
test('both generator and compact workbench expose shared API configuration',()=>{
  for(const file of ['fullstack.js','image_generator.js']){const text=fs.readFileSync(path.join(__dirname,'../web',file),'utf8');assert.match(text,/import \{ openAPISettings \} from '\.\/api_settings.js'/);assert.match(text,/openAPISettings\(this.state,this.cfg/);}
});

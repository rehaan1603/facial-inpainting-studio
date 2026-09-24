// Synthetic DOM tests exercise asynchronous client state without photos, a browser or GPU.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const framing = require('../webapp/dist/framing.js');
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
const response = (body, status = 200) => ({ok: status < 400, status, json: async () => body});
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness(file, fetcher) {
  const elements = new Map(), urls = new Map(); let serial = 0;
  class Pixels { constructor(width, height) { this.width=width; this.height=height; this.data=new Uint8ClampedArray(width*height*4); } }
  class Element {
    constructor(id='', tag='input') { Object.assign(this,{id,tag,width:512,height:512,disabled:false,hidden:false,checked:false,value:'',files:[],style:{},children:[],listeners:{},textContent:''}); this.classList={add(){},remove(){},toggle(){}}; }
    addEventListener(type,fn){(this.listeners[type]??=[]).push(fn);}
    async fire(type,event={}){event.target??=this;event.type=type;for(const fn of this.listeners[type]||[])await fn(event);if(this['on'+type])await this['on'+type](event);}
    dispatchEvent(event){return this.fire(event.type,event);}
    removeAttribute(name){delete this[name];}
    setAttribute(name,value){this[name]=value;}
    replaceChildren(...items){this.children=items;}
    append(...items){this.children.push(...items);}
    querySelector(){return element('learnedOption');}
    getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height};}
    setPointerCapture(){} focus(){} click(){return this.fire('click');}
    toDataURL(){return 'data:image/png;base64,'+(this.context?.marker||'synthetic');}
    getContext(){
      if(this.context)return this.context;
      const canvas=this;
      return this.context={marker:'',pixels:null,drawImage(image){this.marker=image.marker||image.name||'';this.pixels=image.pixels?.slice()||null;},clearRect(){this.pixels=null;},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){},
        createImageData(w,h){return new Pixels(w,h);},putImageData(image){this.pixels=image.data.slice();},
        getImageData(x,y,w,h){const image=new Pixels(w,h);if(this.pixels?.length===image.data.length)image.data.set(this.pixels);else for(let i=3;i<image.data.length;i+=4)image.data[i]=255;return image;}};
    }
  }
  function element(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);}
  const bitmap = file => ({width:file.width||4,height:file.height||4,naturalWidth:file.width||4,naturalHeight:file.height||4,pixels:file.pixels,marker:file.name,close(){}});
  class Image {
    set src(url){const file=urls.get(url)||{name:url};Promise.resolve(file.wait).then(()=>{Object.assign(this,bitmap(file));this.onload?.();});}
  }
  const calls=[];
  const sandbox={console,Math,Uint8Array,Uint8ClampedArray,Float32Array,ImageData:Pixels,Image,Blob,atob,
    URL:{createObjectURL(file){const url='blob:synthetic-'+(++serial);urls.set(url,file);return url;},revokeObjectURL(){}},
    document:{getElementById:element,createElement:tag=>new Element('',tag),querySelectorAll:()=>[...elements.values()]},window:{addEventListener(){}},
    createImageBitmap:async file=>{await file.wait;return bitmap(file);},AbortSignal:{timeout:()=>({})},Event:class {constructor(type){this.type=type;}},
    setTimeout:fn=>{queueMicrotask(fn);return 1;},fetch:async(...args)=>{calls.push(args);return (fetcher||(()=>response({sample_available:false,token:'test'})))(...args);},InpaintingFraming:framing};
  element('method').value='reference';element('strength').value='.99';element('framing').value='crop';element('backbone').value='lama';element('mode').value='painted';element('detail').value='standard';
  const context=vm.createContext(sandbox);vm.runInContext(fs.readFileSync(path.join(__dirname,'../webapp/dist',file),'utf8'),context);
  return {element,calls,eval:source=>vm.runInContext(source,context),setFetch:fn=>{fetcher=fn;},set(name,value){context[name]=value;}};
}
const image = (name, extra={}) => ({name,size:64,width:4,height:4,...extra});
async function upload(h,id,files){h.element(id).files=files;await h.element(id).fire('change');}
async function confidenceTests(){
  const h=harness('confidence.js');await upload(h,'photo',[image('A')]);
  h.eval("references=['A1','A2','A3'];ready()");assert.equal(h.element('reconstruct').disabled,false);
  await upload(h,'photo',[image('B')]);assert.equal(h.eval('references.length'),0);assert.equal(h.element('reconstruct').disabled,true);
  const refs=deferred();const pendingRefs=upload(h,'references',[image('Aref',{wait:refs.promise}),image('r2'),image('r3')]);
  assert.equal(h.element('reconstruct').disabled,true);await upload(h,'photo',[image('C')]);refs.resolve();await pendingRefs;
  assert.equal(h.eval('references.length'),0,'Old references must not attach to a new target');
  const map=deferred(), black=new Uint8ClampedArray(64);for(let i=3;i<64;i+=4)black[i]=255;
  const pendingMap=upload(h,'mapFile',[image('old-map',{wait:map.promise,pixels:black})]);await upload(h,'photo',[image('D')]);map.resolve();await pendingMap;
  assert.equal(h.eval('confidence.every(v=>v===255)'),true,'Old maps must not modify a new target');
  const oldPhoto=deferred(),pendingPhoto=upload(h,'photo',[image('old',{wait:oldPhoto.promise,width:8})]);await upload(h,'photo',[image('latest',{width:6})]);oldPhoto.resolve();await pendingPhoto;
  assert.equal(h.element('editor').width,6,'The last requested target must win');
  await upload(h,'photo',[image('map-target')]);
  await upload(h,'mapFile',[image('transparent',{pixels:new Uint8ClampedArray(64)})]);
  assert.match(h.element('mapStatus').textContent,/opaque.*without transparency/);assert.equal(h.eval('confidence.every(v=>v===255)'),true);
  await upload(h,'mapFile',[image('opaque',{pixels:black})]);assert.equal(h.eval('confidence.every(v=>v===0)'),true);
  h.eval("references=['r1','r2','r3'];ready()");h.element('strength').value='.5';const before=h.calls.length;await h.element('reconstruct').fire('click');
  assert.match(h.element('mapStatus').textContent,/require Strong/);assert.equal(h.calls.length,before,'Missing areas with gentle strength must not start inference');
  h.element('resultPanel').hidden=false;h.element('strength').value='.99';await h.element('strength').fire('change');assert.equal(h.element('resultPanel').hidden,true);assert.match(h.element('mapStatus').textContent,/changed/);
  const sample=deferred();h.setFetch(()=>sample.promise);const pendingSample=h.element('sample').fire('click');await upload(h,'photo',[image('client',{width:7})]);
  sample.resolve(response({image:'data:image/png;base64,AA==',references:['sample1','sample2','sample3']}));await pendingSample;
  assert.equal(h.element('editor').width,7);assert.equal(h.eval('references.length'),0,'Late samples must not overwrite client targets');
  let polls=0;h.setFetch(()=>{polls++;if(polls<3)throw Error('temporary');return response({status:'complete'});});
  await h.eval("readJob('same-job')");assert.equal(polls,3,'Transient polling errors must retry');
  h.eval("currentJob='persistent-job'");h.setFetch(()=>{throw Error('offline');});await h.element('reconstruct').fire('click');
  assert.equal(h.eval('currentJob'),'persistent-job');assert.equal(h.element('photo').disabled,true);assert.equal(h.element('reconstruct').disabled,false);
  assert.match(h.element('reconstruct').textContent,/Reconnect/);
  const resumeCalls=[];h.setFetch(url=>{resumeCalls.push(url);return response({status:'complete',result:'/result.png',metadata:'/settings.json'});});await h.element('reconstruct').fire('click');
  assert.deepEqual(resumeCalls,['/api/jobs/persistent-job'],'Reconnect must poll the same job without POSTing again');assert.equal(h.eval('currentJob'),null);assert.equal(h.element('resultPanel').hidden,false);
}
async function mainTests(){
  const h=harness('app.js');await flush();h.set('input',image('A'));await h.eval('loadImage(input)');
  const refs=deferred();const pending=upload(h,'referenceFiles',[image('Aref',{wait:refs.promise})]);
  h.set('input',image('B'));await h.eval('loadImage(input)');refs.resolve();await pending;assert.equal(h.eval('references.length'),0);
  const map=deferred();h.eval('let applyCalls=0;applyMask=()=>{applyCalls++}');const pendingMap=upload(h,'maskFile',[image('oldmask',{wait:map.promise})]);
  h.set('input',image('C'));await h.eval('loadImage(input)');map.resolve();await pendingMap;assert.equal(h.eval('applyCalls'),0);
  const old=deferred();h.set('input',image('old',{wait:old.promise}));const pendingPhoto=h.eval('loadImage(input)');h.set('input',image('new'));await h.eval('loadImage(input)');old.resolve();await pendingPhoto;
  assert.equal(h.element('sourceName').textContent,'new');
  for(const id of ['backbone','mode','blend','detail','neutralize']){h.element('comparison').hidden=false;await h.element(id).fire('change');assert.equal(h.element('comparison').hidden,true,`${id} must clear a stale result`);}
  h.eval("references=[{name:'ref',data:'synthetic'}];renderReferences()");h.element('comparison').hidden=false;
  await h.element('referenceList').children[0].children[1].click();assert.equal(h.element('comparison').hidden,true);
  const gate=deferred();h.setFetch(()=>gate.promise);const demo=h.eval('loadDemo()');h.set('input',image('client'));await h.eval('loadImage(input)');gate.resolve(response({image:'demo',mask:'demo-mask',references:[]}));await demo;
  assert.equal(h.element('sourceName').textContent,'client','Late demo must not replace the client upload');
  const waiting=deferred();h.set('input',image('pending',{wait:waiting.promise}));const load=h.eval('loadImage(input)');assert.equal(h.element('run').disabled,true);
  await assert.rejects(h.eval('reconstruct()'),/Wait for/);waiting.resolve();await load;
}
(async()=>{await confidenceTests();await mainTests();console.log('Studio upload state: target/reference/map races, stale results, transparent maps, strong missing-area guard and reconnect checks passed');})().catch(error=>{console.error(error);process.exitCode=1;});

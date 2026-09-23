'use strict';
const el=id=>document.getElementById(id), canvas=el('editor'), context=canvas.getContext('2d');
let photo=null, confidence=null, history=[],painting=false,last=null;
function notify(text){el('mapStatus').textContent=text;}
function checkpoint(){history.push(confidence.slice());if(history.length>12)history.shift();el('undo').disabled=false;}
function render(){
 if(!photo)return;context.putImageData(photo,0,0);
 if(el('overlayToggle').checked){const overlay=context.getImageData(0,0,canvas.width,canvas.height);
  for(let i=0;i<confidence.length;i++){const c=confidence[i];if(c===255)continue;const a=(1-c/255)*.55;const color=c===0?[225,54,61]:[237,164,40];for(let j=0;j<3;j++)overlay.data[4*i+j]=Math.round(overlay.data[4*i+j]*(1-a)+color[j]*a);}
  context.putImageData(overlay,0,0);
 }
}
function position(event){const b=canvas.getBoundingClientRect();return [(event.clientX-b.left)*canvas.width/b.width,(event.clientY-b.top)*canvas.height/b.height];}
function dot(x,y){const radius=Number(el('size').value)/2,value=Number(el('evidence').value);for(let yy=Math.max(0,Math.floor(y-radius));yy<Math.min(canvas.height,Math.ceil(y+radius));yy++)for(let xx=Math.max(0,Math.floor(x-radius));xx<Math.min(canvas.width,Math.ceil(x+radius));xx++)if((xx-x)**2+(yy-y)**2<=radius**2)confidence[yy*canvas.width+xx]=value;}
canvas.addEventListener('pointerdown',e=>{if(!photo)return;checkpoint();painting=true;canvas.setPointerCapture(e.pointerId);last=position(e);dot(...last);render();});
canvas.addEventListener('pointermove',e=>{if(!painting)return;const next=position(e),n=Math.max(1,Math.ceil(Math.hypot(next[0]-last[0],next[1]-last[1])/3));for(let i=1;i<=n;i++)dot(last[0]+(next[0]-last[0])*i/n,last[1]+(next[1]-last[1])*i/n);last=next;render();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{painting=false;last=null;});
async function load(file){if(!file)throw Error('Choose a file.');if(file.size>25e6)throw Error('Choose a smaller image.');return createImageBitmap(file);}
el('photo').addEventListener('change',async()=>{try{const image=await load(el('photo').files[0]);if(image.width*image.height>16000000){image.close();throw Error('Use an image below 16 megapixels.');}const ratio=Math.min(1,1024/Math.max(image.width,image.height));canvas.width=Math.round(image.width*ratio);canvas.height=Math.round(image.height*ratio);context.drawImage(image,0,0,canvas.width,canvas.height);image.close();photo=context.getImageData(0,0,canvas.width,canvas.height);confidence=new Uint8Array(canvas.width*canvas.height).fill(255);history=[];for(const id of ['suggest','reset','export'])el(id).disabled=false;el('undo').disabled=true;render();notify('Paint areas that are missing or damaged. Unmarked pixels stay reliable.');}catch(e){notify(e.message);}});
el('mapFile').addEventListener('change',async()=>{try{if(!photo)throw Error('Choose the matching image first.');const image=await load(el('mapFile').files[0]);if(image.width!==canvas.width||image.height!==canvas.height){image.close();throw Error('Map dimensions must match the displayed image.');}const off=document.createElement('canvas');off.width=canvas.width;off.height=canvas.height;const ctx=off.getContext('2d');ctx.drawImage(image,0,0);image.close();const data=ctx.getImageData(0,0,off.width,off.height).data;checkpoint();for(let i=0;i<confidence.length;i++)confidence[i]=Math.round((data[4*i]+data[4*i+1]+data[4*i+2])/3);render();notify('Saved map loaded. White keeps pixels; black replaces them.');}catch(e){notify(e.message);}});
el('suggest').addEventListener('click',()=>{if(!photo)return;if(!confidence.some(v=>v>0&&v<255)){notify('Paint a partially damaged area first. Missing and reliable areas are left unchanged.');return;}checkpoint();const w=canvas.width,h=canvas.height,gray=new Float32Array(w*h);for(let i=0;i<gray.length;i++)gray[i]=(.2126*photo.data[4*i]+.7152*photo.data[4*i+1]+.0722*photo.data[4*i+2])/255;
 for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x;if(confidence[i]===0||confidence[i]===255)continue;const gradient=Math.abs(gray[i+1]-gray[i-1])+Math.abs(gray[i+w]-gray[i-w]);const lap=Math.abs(4*gray[i]-gray[i-1]-gray[i+1]-gray[i-w]-gray[i+w]);const texture=Math.min(1,gradient/.15),noise=Math.min(1,lap/(gradient+.06));confidence[i]=Math.round(255*(.25+.5*texture*(1-.5*noise)));}
 render();notify('Heuristic suggestions applied only to partial areas. Inspect smooth skin, edges and compressed textures; repaint any mistakes.');});
el('undo').addEventListener('click',()=>{if(history.length){confidence=history.pop();el('undo').disabled=!history.length;render();}});
el('reset').addEventListener('click',()=>{if(confidence){checkpoint();confidence.fill(255);render();}});
el('overlayToggle').addEventListener('change',render);
function download(name,data){const off=document.createElement('canvas');off.width=canvas.width;off.height=canvas.height;off.getContext('2d').putImageData(data,0,0);const a=document.createElement('a');a.href=off.toDataURL('image/png');a.download=name;a.click();}
el('export').addEventListener('click',()=>{if(!photo)return;const map=new ImageData(canvas.width,canvas.height),mask=new ImageData(canvas.width,canvas.height);for(let i=0;i<confidence.length;i++){for(let j=0;j<3;j++){map.data[4*i+j]=confidence[i];mask.data[4*i+j]=confidence[i]<255?255:0;}map.data[4*i+3]=mask.data[4*i+3]=255;}download('observed.png',photo);download('damage-mask.png',mask);download('evidence-confidence.png',map);notify('Exported matching image, damage mask and confidence map. Your browser may ask to allow multiple downloads.');});

el('sample').addEventListener('click',async()=>{try{const response=await fetch('/api/demo?reference=1');const data=await response.json();if(!response.ok)throw Error(data.error||'Sample unavailable.');const image=await createImageBitmap(new Blob([Uint8Array.from(atob(data.image.split(',')[1]),c=>c.charCodeAt(0))],{type:'image/png'}));canvas.width=image.width;canvas.height=image.height;context.drawImage(image,0,0);image.close();photo=context.getImageData(0,0,canvas.width,canvas.height);confidence=new Uint8Array(canvas.width*canvas.height).fill(255);history=[];for(const id of ['suggest','reset','export'])el(id).disabled=false;el('undo').disabled=true;render();notify('Local research sample loaded. Mark missing and partially damaged areas separately.');}catch(e){notify(e.message);}});

let references=[],running=false;
function pngData(data){const c=document.createElement('canvas');c.width=data.width;c.height=data.height;c.getContext('2d').putImageData(data,0,0);return c.toDataURL('image/png');}
el('references').addEventListener('change',async()=>{try{const files=[...el('references').files];if(files.length<3||files.length>4)throw Error('Select 3 or 4 reference photos together.');const next=[];for(const file of files){const im=await load(file);if(im.width*im.height>16000000){im.close();throw Error('Use reference photos below 16 megapixels.');}const c=document.createElement('canvas'),ratio=Math.min(1,512/Math.max(im.width,im.height));c.width=Math.round(im.width*ratio);c.height=Math.round(im.height*ratio);c.getContext('2d').drawImage(im,0,0,c.width,c.height);im.close();next.push(c.toDataURL('image/png'));}references=next;el('referenceStatus').textContent=`${next.length} reference photos ready.`;el('reconstruct').disabled=false;}catch(error){references=[];el('reconstruct').disabled=true;el('referenceStatus').textContent=error.message;}});
el('reconstruct').addEventListener('click',async()=>{
 if(running)return;
 try{
  if(!photo||!confidence.some(v=>v<255))throw Error('Load an image and mark at least one damaged area.');
  if(references.length<3)throw Error('Add 3 or 4 matching reference photos.');
  const map=new ImageData(canvas.width,canvas.height),mask=new ImageData(canvas.width,canvas.height);
  for(let i=0;i<confidence.length;i++){for(let k=0;k<3;k++){map.data[4*i+k]=confidence[i];mask.data[4*i+k]=confidence[i]<255?255:0;}map.data[4*i+3]=mask.data[4*i+3]=255;}
  const payload={image:pngData(photo),mask:pngData(mask),confidence:pngData(map),references:[...references],backbone:el('method').value,mode:'painted',detail:'standard',blend:'poisson',strength:Number(el('strength').value)};
  running=true;const controls=[...document.querySelectorAll('button,input,select')];const previous=controls.map(e=>e.disabled);controls.forEach(e=>e.disabled=true);canvas.style.pointerEvents='none';el('resultPanel').hidden=true;
  try{
   notify('Preparing reconstruction…');const sr=await fetch('/api/session'),session=await sr.json();if(!sr.ok)throw Error('Local studio unavailable.');
   const response=await fetch('/api/inpaint',{method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':session.token},body:JSON.stringify(payload)});const job=await response.json();if(!response.ok)throw Error(job.error||'Could not start reconstruction.');
   for(;;){await new Promise(resolve=>setTimeout(resolve,1000));const r=await fetch('/api/jobs/'+job.id),state=await r.json();if(!r.ok)throw Error(state.error||'Could not read reconstruction progress.');notify(state.message);if(state.status==='error')throw Error(state.message);if(state.status==='complete'){el('resultImage').src=state.result;el('resultDownload').href=state.result;el('metadataDownload').href=state.metadata;el('resultPanel').hidden=false;notify('Reconstruction ready. Compare identity and expression carefully; preserved evidence may retain degradation.');break;}}
  }finally{controls.forEach((e,i)=>e.disabled=previous[i]);canvas.style.pointerEvents='';}
 }catch(error){notify(error.message);}finally{running=false;}
});

el('method').addEventListener('change',()=>{el('strength').disabled=el('method').value==='refldm';});

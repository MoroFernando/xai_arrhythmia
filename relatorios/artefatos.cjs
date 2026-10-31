// Leitura independente de ZIP/NPY e agregacao de resultados existentes.
// Nao importa codigo Python, nao carrega modelos e nao calcula mapas ou IoU.
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const crypto = require('crypto');
const BASE = path.resolve(__dirname, '..');
function local(p) {
  const target = path.resolve(BASE, p.replace(/^xai_arrhythmia\//, ''));
  if (target !== BASE && !target.startsWith(BASE + path.sep)) throw Error('Fora de pos3: ' + p);
  return target;
}
const read = p => fs.readFileSync(local(p));
const json = p => JSON.parse(read(p).toString('utf8'));
const hash = p => crypto.createHash('sha256').update(read(p)).digest('hex');
function write(p, value) { const dest = local(p); fs.mkdirSync(path.dirname(dest), {recursive:true}); fs.writeFileSync(dest, value); }
function walk(p = '') { return fs.readdirSync(local(p), {withFileTypes:true}).flatMap(e => e.name === '.git' ? [] : e.isDirectory() ? walk(path.join(p,e.name)) : [path.join(p,e.name).replaceAll('\\','/')]); }
function zip(p) {
  const b = read(p); let e = b.length - 22;
  while (e >= 0 && b.readUInt32LE(e) !== 0x06054b50) e--;
  if(e < 0) throw Error('ZIP invalido: '+p);
  let q = b.readUInt32LE(e+16); const n = b.readUInt16LE(e+10), out = {};
  for(let i=0;i<n;i++) {
    const method=b.readUInt16LE(q+10),size=b.readUInt32LE(q+20),nl=b.readUInt16LE(q+28),el=b.readUInt16LE(q+30),cl=b.readUInt16LE(q+32),lp=b.readUInt32LE(q+42);
    const name=b.subarray(q+46,q+46+nl).toString(),st=lp+30+b.readUInt16LE(lp+26)+b.readUInt16LE(lp+28);
    out[name]=()=> method===8 ? zlib.inflateRawSync(b.subarray(st,st+size)) : b.subarray(st,st+size);
    q+=46+nl+el+cl;
  }
  return out;
}
function npy(b) {
  const v=b[6], len=v===1?b.readUInt16LE(8):b.readUInt32LE(8),start=v===1?10:12,h=b.subarray(start,start+len).toString();
  const dt=/'descr':\s*'([^']+)'/.exec(h)[1],d=b.subarray(start+len),w=+dt.slice(2),out=[];
  if(dt[1]==='O') throw Error('Objetos pickle nao sao permitidos');
  const step=dt[1]==='U'?w*4:w;
  for(let p=0;p<d.length;p+=step) {
    let x;
    if(dt[1]==='U') { x='';for(let j=0;j<w;j++){const c=d.readUInt32LE(p+j*4);if(c)x+=String.fromCodePoint(c);} }
    else if(dt[1]==='S')x=d.subarray(p,p+w).toString().replace(/\0+$/,'');
    else if(dt[1]==='f')x=w===4?d.readFloatLE(p):d.readDoubleLE(p);
    else if(dt[1]==='i')x=w===8?Number(d.readBigInt64LE(p)):w===4?d.readInt32LE(p):d.readInt8(p);
    else if(dt[1]==='u'||dt[1]==='b')x=w===8?Number(d.readBigUInt64LE(p)):w===4?d.readUInt32LE(p):d.readUInt8(p);
    else throw Error('Dtype nao suportado: '+dt);
    out.push(x);
  }
  return out;
}
function npz(p, keys) { const z=zip(p),out={};for(const k of Object.keys(z)){const key=k.replace(/\.npy$/,'');if(!keys||keys.includes(key))out[key]=npy(z[k]());}return out; }
const mean = a => {a=a.filter(Number.isFinite);return a.length?a.reduce((s,x)=>s+x,0)/a.length:null;};
function quant(a,q=.5){a=a.filter(Number.isFinite).slice().sort((a,b)=>a-b);if(!a.length)return null;const p=(a.length-1)*q,i=Math.floor(p);return a[i]+(a[Math.ceil(p)]-a[i])*(p-i);}
function metrics(y,s) {
  const c={tn:0,fp:0,fn:0,tp:0};y.forEach((v,j)=>c[v?(s[j]>=.5?'tp':'fn'):(s[j]>=.5?'fp':'tn')]++);
  const {tn,fp,fn,tp}=c,n=y.length,rec=tp/(tp+fn),spec=tn/(tn+fp),rank=s.map((v,j)=>[v,y[j]]).sort((a,b)=>a[0]-b[0]);let sum=0;
  for(let i=0;i<n;){let j=i+1;while(j<n&&rank[j][0]===rank[i][0])j++;for(let k=i;k<j;k++)if(rank[k][1])sum+=(i+1+j)/2;i=j;}
  return {...c,n,threshold:.5,acuracia:(tp+tn)/n,acuracia_balanceada:(rec+spec)/2,precisao:tp+fp?tp/(tp+fp):0,sensibilidade:rec,especificidade:spec,f1:2*tp/(2*tp+fp+fn),auc_roc:tp+fn&&tn+fp?(sum-(tp+fn)*(tp+fn+1)/2)/((tp+fn)*(tn+fp)):null};
}
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const P='';
const slugs=['shared-extrasystoles-v1','shared-extrasystoles-bbb-v1'];
function loadRuns(){return slugs.map((slug,j)=>{const dir=P+'outputs/models/cnn1d_mitbih_'+slug+'/';return {slug,name:j?'COM':'SEM',dir,m:json(dir+'metrics.json'),t:json(dir+'teste_ludb/ludb_metrics.json'),f:json(dir+'fidelidade_ludb/ludb_fidelity.json'),history:read(dir+'history.csv').toString().trim().split(/\r?\n/).map(s=>s.split(',')),p:npz(dir+'teste_ludb/ludb_predictions.npz'),iou:npz(dir+'fidelidade_ludb/ludb_fidelity.npz')};});}
module.exports={BASE,local,read,json,hash,write,walk,zip,npz,mean,quant,metrics,same,esc,P,slugs,loadRuns};

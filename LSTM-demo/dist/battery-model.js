(function(root){
'use strict';
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const zeros=n=>Array(n).fill(0),mat=(n,m)=>Array.from({length:n},()=>zeros(m));
function config(overrides={}){return {units:1,gap:30,rate:.04,epochs:80,tolerance:.08,patience:12,minDelta:1e-6,clip:1,recurrent:.55,forgetBias:4,threshold:.65,...overrides};}
function sequence(cfg,warning=true){return [[warning?1:0,warning?0:1,0],...Array.from({length:cfg.gap},()=>[0,1,0]),[0,0,1]];}
function create(kind,cfg){const H=cfg.units,p={v:zeros(H).map(()=>8/H),bv:-1};
 const keys=kind==='RNN'?['r']:['f','i','g','o'];
 for(const key of keys){p['W'+key]=mat(H,3);p['U'+key]=mat(H,H);p['b'+key]=zeros(H);
  for(let j=0;j<H;j++){
   if(key==='r'){p.Wr[j][0]=2+.1*j;p.Ur[j][j]=cfg.recurrent;}
   if(key==='f')p.bf[j]=cfg.forgetBias;
   if(key==='i'){p.Wi[j][0]=8;p.bi[j]=-4;}
   if(key==='g')p.Wg[j][0]=2+.1*j;
   if(key==='o'){p.Wo[j][2]=8;p.bo[j]=-4;}
  }
 }
 return {kind,H,p,epoch:0,history:[],status:'Ready',stable:0,lastLoss:null};}
function affine(W,U,b,x,h){return W.map((row,j)=>b[j]+row.reduce((s,w,k)=>s+w*x[k],0)+U[j].reduce((s,w,k)=>s+w*h[k],0));}
function forward(model,xs,y){let h=zeros(model.H),c=zeros(model.H);const cache=[],p=model.p;
 for(let t=0;t<xs.length;t++){const x=xs[t],hp=h.slice(),cp=c.slice(),z={};
  if(model.kind==='RNN'){z.r=affine(p.Wr,p.Ur,p.br,x,hp);h=z.r.map(Math.tanh);cache.push({t,x,hp,cp,z,h:h.slice(),c:zeros(model.H)});}
  else{for(const k of ['f','i','g','o'])z[k]=affine(p['W'+k],p['U'+k],p['b'+k],x,hp);const f=z.f.map(sigmoid),i=z.i.map(sigmoid),g=z.g.map(Math.tanh),o=z.o.map(sigmoid);c=c.map((v,j)=>f[j]*cp[j]+i[j]*g[j]);h=c.map((v,j)=>o[j]*Math.tanh(v));cache.push({t,x,hp,cp,z,f,i,g,o,c:c.slice(),h:h.slice()});}
 }
 const logit=p.bv+p.v.reduce((s,w,j)=>s+w*h[j],0),prob=sigmoid(logit),loss=Math.max(logit,0)-logit*y+Math.log1p(Math.exp(-Math.abs(logit)));
 return {cache,logit,prob,loss,y};}
function blankGrad(p){const out={};for(const [key,value]of Object.entries(p))out[key]=typeof value==='number'?0:Array.isArray(value[0])?value.map(row=>row.map(()=>0)):value.map(()=>0);return out;}
function addOuter(target,a,b){for(let j=0;j<a.length;j++)for(let k=0;k<b.length;k++)target[j][k]+=a[j]*b[k];}
function norm(a){return Math.sqrt(a.reduce((s,v)=>s+v*v,0));}
function backward(model,trace){const p=model.p,H=model.H,grad=blankGrad(p),dzOut=trace.prob-trace.y,T=trace.cache.length;grad.bv=dzOut;grad.v=trace.cache[T-1].h.map(v=>dzOut*v);let dh=p.v.map(v=>dzOut*v),dc=zeros(H);const steps=[];
 for(let t=T-1;t>=0;t--){const d=trace.cache[t],incoming=dh.slice(),dz={},dcTotal=zeros(H),dhp=zeros(H),dx=zeros(3);let dcp=zeros(H);
  if(model.kind==='RNN'){dz.r=dh.map((v,j)=>v*(1-d.h[j]*d.h[j]));}
  else{for(const k of ['f','i','g','o'])dz[k]=zeros(H);
   for(let j=0;j<H;j++){const tc=Math.tanh(d.c[j]);dcTotal[j]=dc[j]+dh[j]*d.o[j]*(1-tc*tc);dz.o[j]=dh[j]*tc*d.o[j]*(1-d.o[j]);dz.f[j]=dcTotal[j]*d.cp[j]*d.f[j]*(1-d.f[j]);dz.i[j]=dcTotal[j]*d.g[j]*d.i[j]*(1-d.i[j]);dz.g[j]=dcTotal[j]*d.i[j]*(1-d.g[j]*d.g[j]);dcp[j]=dcTotal[j]*d.f[j];}
  }
  for(const key of Object.keys(dz)){const delta=dz[key];addOuter(grad['W'+key],delta,d.x);addOuter(grad['U'+key],delta,d.hp);for(let j=0;j<H;j++){grad['b'+key][j]+=delta[j];for(let k=0;k<H;k++)dhp[k]+=p['U'+key][j][k]*delta[j];for(let k=0;k<3;k++)dx[k]+=p['W'+key][j][k]*delta[j];}}
  steps[t]={t,dh:incoming,dc:dcTotal,dz,dhp,dcp,dx,dhNorm:norm(incoming),dcNorm:norm(dcTotal)};dh=dhp;dc=dcp;
 }
 return {grad,steps,dzOut,norm:gradientNorm(grad)};}
function entries(p){const out=[];for(const [key,value]of Object.entries(p)){if(typeof value==='number')out.push({key,indices:[],value});else if(Array.isArray(value[0]))value.forEach((row,j)=>row.forEach((v,k)=>out.push({key,indices:[j,k],value:v})));else value.forEach((v,j)=>out.push({key,indices:[j],value:v}));}return out;}
function valueAt(p,item){return item.indices.reduce((v,i)=>v[i],p[item.key]);}
function setAt(p,item,value){if(!item.indices.length)p[item.key]=value;else if(item.indices.length===1)p[item.key][item.indices[0]]=value;else p[item.key][item.indices[0]][item.indices[1]]=value;}
function gradientNorm(g){return Math.hypot(...entries(g).map(item=>item.value));}
function batch(model,cfg){const grad=blankGrad(model.p),traces=[],backs=[];let loss=0;
 for(const warning of [false,true]){const trace=forward(model,sequence(cfg,warning),warning?1:0),back=backward(model,trace);loss+=trace.loss/2;traces.push(trace);backs.push(back);for(const item of entries(grad))setAt(grad,item,valueAt(grad,item)+valueAt(back.grad,item)/2);}
 return {grad,loss,norm:gradientNorm(grad),traces,backs};}
function apply(model,result,cfg){const factor=cfg.clip>0&&result.norm>cfg.clip?cfg.clip/result.norm:1,before=JSON.parse(JSON.stringify(model.p));for(const item of entries(model.p))setAt(model.p,item,item.value-cfg.rate*factor*valueAt(result.grad,item));return {before,after:JSON.parse(JSON.stringify(model.p)),grad:result.grad,factor,norm:result.norm};}
function epoch(model,cfg){const b=batch(model,cfg),update=apply(model,b,cfg),after=batch(model,cfg),warning=after.traces[1].prob,healthy=after.traces[0].prob;model.epoch++;model.history.push({epoch:model.epoch,loss:after.loss,warning,healthy,norm:b.norm});
 const change=model.lastLoss===null?Infinity:Math.abs(model.lastLoss-after.loss);model.stable=change<cfg.minDelta?model.stable+1:0;model.lastLoss=after.loss;
 if(!Number.isFinite(after.loss)||entries(model.p).some(item=>!Number.isFinite(item.value)))model.status='Numerical divergence';else if(after.loss<=cfg.tolerance)model.status='Loss target met';else if(model.stable>=cfg.patience)model.status='Plateau above target';else model.status='Training';return {batch:b,update,after};}
const api={config,sequence,create,forward,backward,batch,apply,epoch,entries,valueAt,setAt,gradientNorm,sigmoid};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.BatteryModel=api;
})(typeof window!=='undefined'?window:globalThis);

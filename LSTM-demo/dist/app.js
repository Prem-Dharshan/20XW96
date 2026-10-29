(() => {
'use strict';
const $=id=>document.getElementById(id), svg=$('network'),world=$('world'),ns='http://www.w3.org/2000/svg';
const sketch=rough.svg(svg);
const palette=()=>{const style=getComputedStyle(document.documentElement);return Object.fromEntries(['memory','hidden','input','text','accent'].map(key=>[key,style.getPropertyValue('--'+key).trim()]));};
function stableSeed(value){let seed=7;for(const c of value)seed=(seed*31+c.charCodeAt(0))%2147483647;return seed||7;}
const weights={f:[-3,-3,4,4],i:[3,3,-4,-4],g:[1.5,-1.5,0,0],o:[-3,-3,-3,3]},U={f:.5,i:-.5,g:0,o:.5};
const sigmoid=z=>1/(1+Math.exp(-z)),n=v=>v.toFixed(4),names=['Input & previous states','Forget gate','Input gate','Candidate memory','Retain & write','Update memory','Output gate','Expose hidden state','Pass forward'];
let color='RED',cells=[],step=0,playing=false,timer=null,raf=null,animationId=0,view='focus',camera={x:0,y:0,k:1},drag=null;
let steps=[];const expanded=[true,true,false,false],offsets=[];
const token=t=>t===0?color:t===3?'ASK':'blank';
function rebuildSteps(t=0,p=0){steps=[];let x=40;for(let j=0;j<4;j++){offsets[j]=x;x+=expanded[j]?550:250;if(expanded[j])for(let phase=0;phase<9;phase++)steps.push({t:j,p:phase});else steps.push({t:j,p:9});}steps.push({t:3,p:10});step=Math.max(0,steps.findIndex(s=>s.t===t&&(p===10?s.p===10:expanded[t]?s.p===(p===9?0:p):s.p===9)));}
function jumpPhase(t,p){const index=steps.findIndex(s=>s.t===t&&s.p===p);if(index>=0)go(index);}
function compute(){let c=0,h=0;const xs=[color==='RED'?[1,0,0,0]:[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]];cells=xs.map((x,t)=>{const cp=c,hp=h,z={},dot={};for(const key of Object.keys(weights)){dot[key]=weights[key].reduce((s,v,j)=>s+v*x[j],0);z[key]=dot[key]+U[key]*hp;}const f=sigmoid(z.f),i=sigmoid(z.i),g=Math.tanh(z.g),o=sigmoid(z.o),keep=f*cp,write=i*g;c=keep+write;h=o*Math.tanh(c);return {t,x,cp,hp,z,dot,f,i,g,o,keep,write,c,h};});}
function element(tag,attrs={},text=''){
 const cls=attrs.class||'';
 if((tag==='rect'||tag==='circle')&&(/node|cell-outline|op/.test(cls))){
  const colors=palette(),active=cls.includes('active'),cell=cls.includes('cell-outline');
  const options={seed:stableSeed(JSON.stringify(attrs)),roughness:1.35,bowing:1.1,stroke:active?colors.accent:colors.text,strokeWidth:cell?1.3:1.5,fill:active&&!cell?'#d6e5eb':'#fffefa',fillStyle:active&&!cell?'hachure':'solid',hachureGap:5,fillWeight:.5};
  const el=tag==='rect'?sketch.rectangle(Number(attrs.x),Number(attrs.y),Number(attrs.width),Number(attrs.height),options):sketch.circle(Number(attrs.cx),Number(attrs.cy),Number(attrs.r)*2,options);
  for(const [key,value]of Object.entries(attrs))if(!['x','y','width','height','rx','cx','cy','r'].includes(key))el.setAttribute(key,value);
  return el;
 }
 const el=document.createElementNS(ns,tag);for(const [key,value]of Object.entries(attrs))el.setAttribute(key,value);if(text)el.textContent=text;return el;
}
function label(parent,x,y,text,cls='',anchor='middle'){parent.append(element('text',{x,y,class:cls,'text-anchor':anchor},text));}
function wire(parent,d,id,type='',lit=false){
 parent.append(element('path',{d,'data-path':id,class:`wire ${type} ${lit?'lit':''}`}));
 const colors=palette(),stroke=colors[type]||colors.text;
 const drawing=sketch.path(d,{seed:stableSeed(id),roughness:.85,bowing:.8,stroke,strokeWidth:lit?2.3:type==='memory'?1.8:1.5});
 const paths=drawing.querySelectorAll('path');if(paths.length)paths[paths.length-1].setAttribute('marker-end','url(#arrow)');
 drawing.setAttribute('pointer-events','none');parent.append(drawing);
}
function operation(parent,x,y,text,active=false){parent.append(element('circle',{cx:x,cy:y,r:17,class:'op'+(active?' active-node':'')}));label(parent,x,y+6,text);}
function gate(parent,x,y,title,value,key,t,active){const group=element('g',{class:'gate-hit',role:'button','aria-label':`Cell ${t+1}: ${title}`});group.append(element('rect',{x:x-42,y:y-30,width:84,height:60,rx:8,class:'node'+(active?' active-node':'')}));label(group,x,y-6,title);label(group,x,y+17,value,'value');group.addEventListener('click',()=>jumpPhase(t,{f:1,i:2,g:3,o:6}[key]));parent.append(group);}
function drawCell(t){const d=cells[t],s=steps[step],active=t===s.t,p=t<s.t?8:active?s.p:-1,x=offsets[t],group=element('g',{transform:`translate(${x},38)`});world.append(group);
 group.append(element('rect',{x:0,y:0,width:500,height:414,rx:15,class:'cell-outline'+(active?' cell-active':'')}));label(group,20,30,`CELL ${t+1} · ${token(t)}`,'','start');label(group,480,30,'EXPANDED','small','end');
 const gx=[82,194,306,418],gy=246;
 wire(group,'M -25 86 H 65',`${t}-c-in`,'memory',active&&p===0);wire(group,'M 99 86 H 289',`${t}-keep`,'memory',active&&(p===1||p===4));wire(group,'M 323 86 H 525',`${t}-c-out`,'memory',active&&(p===5||p===8));
 wire(group,'M -25 372 H 65',`${t}-h-in`,'hidden',active&&p===0);
 wire(group,'M 82 342 V 302 H 418',`${t}-bus`,'input',active&&p===0);
 wire(group,'M 35 372 V 318 H 418',`${t}-h-bus`,'hidden',active&&p===0);
 wire(group,'M 82 400 V 342',`${t}-x`,'input',active&&p===0);
 for(const [j,key,phase]of [[0,'f',1],[1,'i',2],[2,'g',3],[3,'o',6]]){
  wire(group,`M ${gx[j]} 302 V 278`,`${t}-${key}-in`,'input',active&&p===phase);
  wire(group,`M ${gx[j]} 318 V 278`,`${t}-${key}-h`,'hidden',active&&p===phase);
 }
 wire(group,'M 82 214 V 104',`${t}-f-out`,'',active&&(p===1||p===4));
 wire(group,'M 194 214 V 152 H 288',`${t}-i-out`,'',active&&(p===2||p===4));
 wire(group,'M 306 214 V 170',`${t}-g-out`,'',active&&(p===3||p===4));wire(group,'M 306 134 V 104',`${t}-write`,'memory',active&&(p===4||p===5));
 wire(group,'M 465 86 V 128',`${t}-c-tanh`,'memory',active&&p===7);
 wire(group,'M 465 184 V 372 H 436',`${t}-tanh-out`,'hidden',active&&p===7);
 wire(group,'M 418 278 V 354',`${t}-o-out`,'',active&&(p===6||p===7));wire(group,'M 436 372 H 525',`${t}-h-out`,'hidden',active&&(p===7||p===8));
 operation(group,82,86,'×',active&&p===4);operation(group,306,152,'×',active&&p===4);operation(group,306,86,'+',active&&p===5);operation(group,418,372,'×',active&&p===7);
 group.append(element('rect',{x:432,y:128,width:66,height:56,rx:8,class:'node'+(active&&p===7?' active-node':'')}));label(group,465,151,'tanh');label(group,465,174,p>=7?n(Math.tanh(d.c)):'cₜ','value');
 for(const [j,key,title,phase]of [[0,'f','σ · f',1],[1,'i','σ · i',2],[2,'g','tanh · g',3],[3,'o','σ · o',6]])gate(group,gx[j],gy,title,p>=phase?n(d[key]):`${key}ₜ`,key,t,active&&p===phase);
 label(group,14,65,`c${t} ${n(d.cp)}`,'memory-label value','start');label(group,486,65,`c${t+1} ${p>=5?n(d.c):'—'}`,'memory-label value','end');
 label(group,14,360,`h${t} ${n(d.hp)}`,'hidden-label value','start');label(group,486,405,`h${t+1} ${p>=7?n(d.h):'—'}`,'hidden-label value','end');label(group,82,432,`x${t+1} · ${token(t)}`,'small');
 if(active&&p===4){label(group,166,115,`kept ${n(d.keep)}`,'small');label(group,340,173,`new ${n(d.write)}`,'small','start');}
}
function drawCompact(t){const d=cells[t],s=steps[step],active=s.t===t,done=t<s.t||active,x=offsets[t];const g=element('g',{transform:`translate(${x},38)`});world.append(g);
 wire(g,'M -25 86 H 225',`${t}-c-out`,'memory',active);wire(g,'M -25 372 H 225',`${t}-h-out`,'hidden',active);
 g.append(element('rect',{x:0,y:55,width:200,height:337,rx:15,class:'cell-outline'+(active?' cell-active':'')}));label(g,100,160,`CELL ${t+1}`);label(g,100,192,'LSTM');label(g,100,233,token(t));label(g,100,263,'same weights','small');
 label(g,100,33,`c${t+1} ${done?n(d.c):'—'}`,'memory-label value');label(g,100,427,`h${t+1} ${done?n(d.h):'—'}`,'hidden-label value');
 wire(g,'M 100 475 V 397',`${t}-x`,'input',active);label(g,100,501,`x${t+1} · ${token(t)}`,'small');
}
function draw(){animationId++;cancelAnimationFrame(raf);world.replaceChildren();for(let t=0;t<4;t++)expanded[t]?drawCell(t):drawCompact(t);
 const readoutX=offsets[3]+(expanded[3]?560:260),d=cells[3],g=element('g',{transform:`translate(${readoutX},230)`});world.append(g);wire(g,'M -35 180 V 0 H 20','readout','hidden',steps[step].p===10);g.append(element('rect',{x:20,y:-50,width:140,height:102,rx:12,class:'node'+(steps[step].p===10?' active-node':'')}));label(g,90,-19,'Softmax');label(g,90,12,steps[step].p===10?(d.h>0?'RED':'BLUE'):'Prediction');label(g,90,36,'at ASK only','small');applyCamera();}
function applyCamera(){world.setAttribute('transform',`translate(${camera.x},${camera.y}) scale(${camera.k})`);}
function positionCamera(){const r=svg.getBoundingClientRect(),s=steps[step];if(view==='overview'){const width=offsets[3]+(expanded[3]?750:450);camera.k=Math.min(r.width/width,r.height/560);camera.x=(r.width-width*camera.k)/2+20*camera.k;camera.y=(r.height-560*camera.k)/2;}else{const width=expanded[s.t]?570:310;camera.k=Math.min(r.width/width,r.height/550,1.7);const cx=offsets[s.t]+(expanded[s.t]?250:100);camera.x=r.width/2-cx*camera.k;camera.y=(r.height-550*camera.k)/2;}applyCamera();}
function packetAnimation(){const s=steps[step],t=s.t,p=s.p;const keys=p===0?['c-in','h-in','x','bus','h-bus']:p===1?['f-in','f-h','f-out']:p===2?['i-in','i-h','i-out']:p===3?['g-in','g-out']:p===4?['keep','i-out','g-out','write']:p===5?['write','c-out']:p===6?['o-in','o-h','o-out']:p===7?['c-tanh','tanh-out','o-out','h-out']:['c-out','h-out'];const paths=keys.map(k=>world.querySelector(`[data-path="${t}-${k}"]`)).filter(Boolean);if(p===10){const rp=world.querySelector('[data-path="readout"]');if(rp)paths.push(rp);}
 if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;const colors=palette();const dots=paths.map(path=>{const dot=element('circle',{r:5,fill:path.classList.contains('hidden')?colors.hidden:path.classList.contains('input')?colors.input:colors.memory});path.parentElement.append(dot);return{path,dot,length:path.getTotalLength()};});const id=++animationId,start=performance.now(),duration=Math.min(2200,Number($('speed').value)*.65);
 function tick(time){if(id!==animationId)return;const progress=Math.min(1,(time-start)/duration);for(const item of dots){const pt=item.path.getPointAtLength(item.length*progress);item.dot.setAttribute('cx',pt.x);item.dot.setAttribute('cy',pt.y);}if(progress<1)raf=requestAnimationFrame(tick);else dots.forEach(item=>item.dot.remove());}raf=requestAnimationFrame(tick);}
function stageInfo(){const s=steps[step],d=cells[s.t],prev=`h${s.t}`,t=s.t+1,aff=k=>`${n(d.dot[k])} + (${U[k]}) × ${n(d.hp)} + 0`,red=sigmoid(8*cells[3].h);
const data=[
 ['Read the input','Each gate combines the current token with the previous hidden state. The previous cell state travels along a separate memory lane.',`xₜ = one-hot(tokenₜ)<br>zₐ = Wₐxₜ + Uₐhₜ₋₁ + bₐ`, `x${t} = [${d.x.join(', ')}]<br>${prev} = ${n(d.hp)}<br>c${s.t} = ${n(d.cp)}`,'Initial / incoming memory',n(d.cp),'The gate inputs use hₜ₋₁. No gate uses the newly computed hₜ.'],
 ['Forget gate','How much of the previous memory should survive? A sigmoid produces a fraction between 0 and 1.','fₜ = σ(Wբxₜ + Uբhₜ₋₁ + bբ)',`zբ = ${aff('f')}<br>= ${n(d.z.f)}<br>fₜ = σ(${n(d.z.f)}) = ${n(d.f)}`,'Fraction of old memory retained',n(d.f),`Retains ${(100*d.f).toFixed(2)}% of c${s.t}. ${s.t===0?'The initial memory is zero, so there is nothing to retain.':'The blank input makes this gate almost fully open.'}`],
 ['Input gate','How much candidate content should be written? This gate controls the amount, while the candidate supplies the content.','iₜ = σ(Wᵢxₜ + Uᵢhₜ₋₁ + bᵢ)',`zᵢ = ${aff('i')}<br>= ${n(d.z.i)}<br>iₜ = σ(${n(d.z.i)}) = ${n(d.i)}`,'Fraction of candidate written',n(d.i),`Writes ${(100*d.i).toFixed(2)}% of the candidate. This is not the candidate value itself.`],
 ['Candidate memory','What could we store? A tanh transforms the weighted inputs into proposed content between −1 and 1.','gₜ = c̃ₜ = tanh(Wgxₜ + Ughₜ₋₁ + bg)',`zg = ${aff('g')}<br>= ${n(d.z.g)}<br>gₜ = tanh(${n(d.z.g)}) = ${n(d.g)}`,'Proposed memory content',n(d.g),s.t===0?`${color} proposes ${color==='RED'?'positive':'negative'} memory. The input gate determines how much enters cₜ.`:'A blank proposes zero new content. The old memory can survive without being rewritten.'],
 ['Retain and write','Two separate multiplications: scale the old memory, and scale the proposed new content.','retained = fₜ ⊙ cₜ₋₁<br>new = iₜ ⊙ gₜ',`retained = ${n(d.f)} × ${n(d.cp)}<br>= ${n(d.keep)}<br>new = ${n(d.i)} × ${n(d.g)}<br>= ${n(d.write)}`,'Retained + new',`${n(d.keep)} + ${n(d.write)}`,'⊙ means elementwise multiplication. With one memory unit, it is ordinary multiplication.'],
 ['Update the cell state','Add the retained memory and the accepted new content. This is the memory that travels to the next cell.','cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ',`c${t} = ${n(d.keep)} + ${n(d.write)}<br>= ${n(d.c)}`,'New internal memory cₜ',n(d.c),'The cell state is not passed through tanh on this memory lane. A copy will be squashed for the hidden state.'],
 ['Output gate','How much of the squashed memory should be exposed? The output gate controls hₜ, while cₜ continues along its own lane.','oₜ = σ(Wₒxₜ + Uₒhₜ₋₁ + bₒ)',`zₒ = ${aff('o')}<br>= ${n(d.z.o)}<br>oₜ = σ(${n(d.z.o)}) = ${n(d.o)}`,'Fraction exposed after tanh',n(d.o),`Exposes ${(100*d.o).toFixed(2)}% of tanh(cₜ). Small hₜ does not mean the internal memory cₜ is small.`],
 ['Expose the hidden state','Squash a copy of cₜ with tanh, then multiply by the output gate. The result is the hidden state.','hₜ = oₜ ⊙ tanh(cₜ)',`tanh(c${t}) = ${n(Math.tanh(d.c))}<br>h${t} = ${n(d.o)} × ${n(Math.tanh(d.c))}<br>= ${n(d.h)}`,'Exposed hidden state hₜ',n(d.h),'hₜ goes into the next cell’s gate calculations. A readout can use it; this task reads it only at ASK.'],
 ['Pass both states forward',s.t===3?'The final hidden state goes to the color readout. Earlier time steps passed both states to the next cell.':'The next time step receives both states together with its own new token.',`(c${t}, h${t}) → ${s.t===3?'readout':'cell '+(t+1)}`,`c${t} = ${n(d.c)}<br>h${t} = ${n(d.h)}<br>${s.t===3?'next: color prediction':'next token = '+token(s.t+1)}`,'Memory passed forward',n(d.c),'These are time steps of one LSTM with shared weights, not independently parameterized layers.'],
 [s.t===0?'Store the color':s.t===3?'ASK: expose the memory':'Preserve through a blank',s.t===0?'All gates run in this one cell step. The color proposes signed content and the input gate admits it to memory.':s.t===3?'The ASK token opens the output gate. Internal memory becomes a strong hidden signal for the readout.':'All gates run internally. A blank adds zero candidate content and retains most of the stored color.','fₜ = σ(Wfxₜ + Ufhₜ₋₁ + bf)<br>iₜ = σ(Wᵢxₜ + Uᵢhₜ₋₁ + bᵢ)<br>gₜ = tanh(Wgxₜ + Ughₜ₋₁ + bg)<br>oₜ = σ(Wₒxₜ + Uₒhₜ₋₁ + bₒ)<br>cₜ = fₜcₜ₋₁ + iₜgₜ<br>hₜ = oₜ tanh(cₜ)',`fₜ = ${n(d.f)}; iₜ = ${n(d.i)}<br>gₜ = ${n(d.g)}; oₜ = ${n(d.o)}<br>c${t} = ${n(d.c)}<br>h${t} = ${n(d.h)}`,'Memory after this cell',n(d.c),'Expand this cell to walk through its gates individually. Collapsing changes the presentation, not the calculation.'],
 ['Predict the remembered color','The readout converts the final hidden state into two logits. Softmax converts logits into probabilities.','[ℓRED, ℓBLUE] = [4h₄, −4h₄]<br>P(RED) = e<sup>ℓRED</sup> / (e<sup>ℓRED</sup> + e<sup>ℓBLUE</sup>)',`h₄ = ${n(cells[3].h)}<br>logits = [${n(4*cells[3].h)}, ${n(-4*cells[3].h)}]<br>P(RED) = ${n(red)}<br>P(BLUE) = ${n(1-red)}`,'Predicted color',cells[3].h>0?'RED':'BLUE',`Select the largest probability. Confidence for ${color}: ${(100*(color==='RED'?red:1-red)).toFixed(2)}%. This readout is separate from the LSTM cell.`]
];return data[s.p];}
function teachingDetails(){const s=steps[step],d=cells[s.t];
 const intuitions=['The token tells the cell what just arrived; the previous states tell it what came before.','Think of fₜ as a tap controlling how much old memory flows through.','Think of iₜ as permission to write, not the content being written.','The candidate is a proposed note: positive for RED, negative for BLUE, zero for a blank.','Keep an accepted part of the old note and an accepted part of the new note.','The memory is the sum of what survived and what was accepted.','The output gate is a window onto memory; closing it does not erase the memory.','The hidden state is the visible, bounded version of the internal memory.','Carry memory and an exposed signal into the next time step.','A whole cell performs the same gate operations, grouped into one presentation step.','Turn the sign and strength of the exposed memory into a choice between two colors.'];
 const purposes=['We need a numerical input so the shared weights can combine the token with sequence context.','Selective retention lets the model remove stale content while preserving useful context.','Selective writing prevents irrelevant inputs from overwriting useful memory.','We need proposed content separately from the decision about how much to store.','Multiplying by the gates independently controls retaining and writing.','An additive update lets old memory continue without being completely recomputed.','We control how much memory influences the next cell and the readout.','tanh bounds the exposed content; the output gate then scales how much is visible.','Passing both states lets one set of weights process the sequence in order.','Grouping the operations lets us follow longer sequences without stopping at every gate.','The readout maps the LSTM state into the task’s output categories.'];
 $('intuition').textContent=intuitions[s.p];$('purpose').textContent=purposes[s.p];
 const gateKey={1:'f',2:'i',3:'g',6:'o'}[s.p];
 const paramRow=k=>`<div class="param-row"><strong>${k==='g'?'Candidate g':k+' gate'}</strong><span>W<sub>${k}</sub> = [${weights[k].join(', ')}]<br>U<sub>${k}</sub> = ${U[k]} · b<sub>${k}</sub> = 0</span></div>`;
 let params='';
 if(s.p===10)params='<div class="param-row"><strong>Readout weights</strong><span>Wᵧ = [[4], [−4]] (2 × 1)<br>bᵧ = [0, 0]</span></div>';
 else if(gateKey)params=paramRow(gateKey);
 else if(s.p===0||s.p===9)params=Object.keys(weights).map(paramRow).join('');
 else params='<p class="param-note">No new weights or biases in this operation. It uses gate values and states already calculated.</p>';
 const current=s.p===10?`h₄ = ${n(cells[3].h)}`:gateKey||s.p===0?`xₜ = [${d.x.join(', ')}]<br>hₜ₋₁ = ${n(d.hp)} · cₜ₋₁ = ${n(d.cp)}`:s.p===4||s.p===5?`fₜ = ${n(d.f)} · iₜ = ${n(d.i)}<br>gₜ = ${n(d.g)} · cₜ₋₁ = ${n(d.cp)}`:s.p===7?`oₜ = ${n(d.o)} · cₜ = ${n(d.c)}`:`cₜ = ${n(d.c)} · hₜ = ${n(d.h)}`;
 $('parameters').innerHTML=params+`<div class="param-row"><strong>Current values (not parameters)</strong><span>${current}</span></div><p class="param-note">Weights and biases are hand chosen here; a trained model would learn them. The same parameters are reused at every cell.</p>`;
 const activation=s.p===1||s.p===2||s.p===6?'Gate activation: sigmoid':s.p===3?'Candidate activation: tanh':s.p===7?'Output transform: tanh':s.p===10?'Readout: softmax over 2 classes':'Gate activations: sigmoid · candidate/output transform: tanh';
 $('hyperparameters').innerHTML=`<div class="hyper-grid"><span>Input size</span><strong>4 features</strong><span>Hidden size</span><strong>1 unit</strong><span>LSTM layers</span><strong>1</strong></div><p>${activation}</p><p class="param-note">Task setting: 4 time steps, 2 colors. No training runs here, so learning rate, optimizer, batch size, and epochs do not apply to this forward-pass demo.</p>`;
}
function toggleCell(t){const s=steps[step];stop();expanded[t]=!expanded[t];rebuildSteps(s.t,s.p);render(true,true);}
function formatMath(text){return text.replaceAll('բ','f').replace(/([WUbz])([figo])/g,'$1<sub>$2</sub>');}
function updateUI(){const s=steps[step],d=cells[s.t],info=stageInfo();$('phase-label').textContent=`CELL ${s.t+1} / ${s.p<9?names[s.p].toUpperCase():s.p===9?'COMPACT CELL':'READOUT'}`;$('gate-title').textContent=info[0];$('gate-explanation').textContent=info[1];$('equation').innerHTML=formatMath(info[2]);$('calculation').innerHTML=formatMath(info[3]);$('result-label').textContent=info[4];$('result').textContent=info[5];$('step-note').textContent=info[6];
 teachingDetails();$('mode-label').textContent=`Cell ${s.t+1} · ${expanded[s.t]?'gate-by-gate':'whole cell'}`;$('toggle-cell').textContent=expanded[s.t]?'Collapse this cell':'Expand this cell';$('toggle-cell').setAttribute('aria-expanded',String(expanded[s.t]));
 $('prev-c').textContent=n(d.cp);$('current-c').textContent=s.p>=5?n(d.c):'—';$('current-h').textContent=s.p>=7?n(d.h):'—';$('answer').textContent=s.p===10?(cells[3].h>0?'RED':'BLUE'):'—';$('step-count').textContent=`${step+1} / ${steps.length}`;$('progress-fill').style.width=`${100*(step+1)/steps.length}%`;$('prev').disabled=step===0;$('next').disabled=step===steps.length-1;$('play').textContent=playing?'Ⅱ Pause':'▶ Play';$('announcement').textContent=`Cell ${s.t+1}: ${info[0]}, ${info[4]} ${info[5]}`;
 $('timeline').innerHTML=cells.map((c,t)=>`<div class="time-item"><button data-time="${t}" class="${t===s.t?'active':''}" aria-current="${t===s.t?'step':'false'}">${t+1} · ${token(t)}</button><button class="time-toggle" data-toggle="${t}" aria-expanded="${expanded[t]}" aria-label="${expanded[t]?'Collapse':'Expand'} cell ${t+1}">${expanded[t]?'− Collapse':'+ Expand'}</button></div>`).join('');$('timeline').querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>go(steps.findIndex(s=>s.t===Number(b.dataset.time))));$('timeline').querySelectorAll('[data-toggle]').forEach(b=>b.onclick=()=>toggleCell(Number(b.dataset.toggle)));
 const entries=[['f','Forget gate',1],['i','Input gate',2],['g','Candidate',3],['c','Cell state',5],['o','Output gate',6],['h','Hidden state',7]];$('gate-list').innerHTML=entries.map(([key,label,p])=>`<button data-phase="${p}" class="${s.p===p?'active':''}" ${!expanded[s.t]?'disabled':''}>${label}<span>${s.p>=p?n(d[key]):'—'}</span></button>`).join('');$('gate-list').querySelectorAll('button').forEach(b=>b.onclick=()=>jumpPhase(s.t,Number(b.dataset.phase)));
}
function stop(){playing=false;clearTimeout(timer);}
function render(animate=false,reframe=false){draw();updateUI();if(reframe)positionCamera();if(animate)packetAnimation();}
function go(target){const old=steps[step].t;stop();step=Math.max(0,Math.min(steps.length-1,target));render(true,old!==steps[step].t);}
function schedule(){clearTimeout(timer);if(!playing)return;timer=setTimeout(()=>{if(step===steps.length-1){stop();updateUI();return;}const old=steps[step].t;step++;render(true,old!==steps[step].t);schedule();},Number($('speed').value));}
function togglePlay(){if(playing){stop();animationId++;cancelAnimationFrame(raf);render();}else{if(step===steps.length-1){step=0;render(false,true);}playing=true;updateUI();packetAnimation();schedule();}}
function zoom(factor,cx=svg.clientWidth/2,cy=svg.clientHeight/2){const next=Math.max(.25,Math.min(2.5,camera.k*factor)),ratio=next/camera.k;camera.x=cx-(cx-camera.x)*ratio;camera.y=cy-(cy-camera.y)*ratio;camera.k=next;view='manual';applyCamera();}
svg.addEventListener('wheel',e=>{e.preventDefault();const r=svg.getBoundingClientRect();zoom(e.deltaY<0?1.12:1/1.12,e.clientX-r.left,e.clientY-r.top);},{passive:false});
svg.addEventListener('pointerdown',e=>{if(e.target.closest('.gate-hit'))return;drag={x:e.clientX,y:e.clientY,cx:camera.x,cy:camera.y};svg.setPointerCapture(e.pointerId);svg.classList.add('dragging');});svg.addEventListener('pointermove',e=>{if(!drag)return;view='manual';camera.x=drag.cx+e.clientX-drag.x;camera.y=drag.cy+e.clientY-drag.y;applyCamera();});const endDrag=()=>{drag=null;svg.classList.remove('dragging');};svg.addEventListener('pointerup',endDrag);svg.addEventListener('pointercancel',endDrag);
 $('prev').onclick=()=>go(step-1);$('next').onclick=()=>go(step+1);$('play').onclick=togglePlay;$('restart').onclick=()=>{go(0);view='focus';positionCamera();};$('speed').onchange=()=>{if(playing)schedule();};$('color').onchange=()=>{color=$('color').value;$('color-chip').textContent=color;compute();go(0);};$('zoom-in').onclick=()=>zoom(1.25);$('zoom-out').onclick=()=>zoom(.8);$('fit').onclick=()=>{view='overview';positionCamera();};$('focus').onclick=()=>{view='focus';positionCamera();};
 $('toggle-cell').onclick=()=>toggleCell(steps[step].t);
 $('toggle-setup').onclick=()=>{const closed=document.body.classList.toggle('inputs-collapsed');$('toggle-setup').textContent=closed?'Show inputs':'Hide inputs';$('toggle-setup').setAttribute('aria-expanded',String(!closed));requestAnimationFrame(positionCamera);};
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('announcement').textContent='Use your browser fullscreen command, F11.';}}
$('fullscreen').onclick=fullscreen;document.addEventListener('fullscreenchange',()=>{$('fullscreen').textContent=document.fullscreenElement?'⛶ Exit fullscreen':'⛶ Fullscreen';positionCamera();});
$('weights-btn').onclick=()=>{stop();updateUI();$('weights').showModal();};$('close-weights').onclick=()=>$('weights').close();$('weights').addEventListener('click',e=>{if(e.target===$('weights')){const r=$('weights').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('weights').close();}});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName)||$('weights').open)return;if(e.key==='ArrowRight'){e.preventDefault();go(step+1);}if(e.key==='ArrowLeft'){e.preventDefault();go(step-1);}if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key.toLowerCase()==='f')fullscreen();});
new ResizeObserver(()=>{if(view!=='manual')positionCamera();}).observe(svg);
rebuildSteps();compute();render(false,true);
})();

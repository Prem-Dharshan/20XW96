(() => {
'use strict';
const $=id=>document.getElementById(id),M=BatteryModel,cfg=M.config(),model=M.create('LSTM',cfg);
const beforeBatch=M.batch(model,cfg),before=JSON.parse(JSON.stringify(model.p)),result=M.epoch(model,cfg),afterBatch=result.after;
const glossary={
 x:['xₜ','Current input','The encoded event observed at time t.'],
 h:['hₜ','Hidden state','Exposed output; also context for the next cell’s gates.'],
 hp:['hₜ₋₁','Previous hidden state','Exposed context arriving from the preceding time step.'],
 c:['cₜ','Cell state','Internal memory that travels along the additive memory path.'],
 cp:['cₜ₋₁','Previous cell state','Old internal memory before processing the current input.'],
 f:['fₜ','Forget gate','Fraction of old memory retained; a sigmoid control value.'],
 i:['iₜ','Input gate','Fraction of candidate content accepted into memory.'],
 g:['gₜ','Candidate content','Proposed signed content; also called c̃ₜ.'],
 o:['oₜ','Output gate','Fraction of tanh(memory) exposed as the hidden state.'],
 W:['Wₐ','Input weights','Trainable matrix connecting the current input to component a.'],
 U:['Uₐ','Recurrent weights','Trainable matrix connecting previous hidden context to component a.'],
 b:['bₐ','Gate bias','Trainable offset in a component’s affine preactivation.'],
 z:['zₐ,ₜ','Preactivation','Weighted sum before applying sigmoid or tanh.'],
 sigma:['σ','Sigmoid','Maps a real number to a fraction between zero and one.'],
 tanh:['tanh','Hyperbolic tangent','Maps a real number to signed content between −1 and 1.'],
 had:['⊙','Elementwise multiplication','Multiply matching vector entries; ordinary multiplication for one unit.'],
 r:['r','Readout logit','Unbounded score computed from the final hidden state.'],
 v:['v','Readout weights','Trainable weights mapping hidden units to the final logit.'],
 by:['bᵧ','Readout bias','Trainable offset in the final classification logit.'],
 p:['p','Warning probability','Sigmoid of the readout logit; compared with the decision threshold.'],
 y:['y','Target label','1 means a warning sequence; 0 means a healthy sequence.'],
 L:['L','Loss','Binary cross-entropy measuring error for one sequence.'],
 dh:['h̄ₜ','Hidden-state gradient','Incoming derivative of loss with respect to the hidden state.'],
 dc:['Dₜ','Total cell-state gradient','Combines future memory correction with the current hidden-output route.'],
 dcfuture:['c̄ₜ future','Future memory gradient','Correction arriving through the next cell’s direct memory route.'],
 dcprev:['c̄ₜ₋₁','Previous-memory gradient','Derivative passed along the direct cell-state route to the prior step.'],
 delta:['δₐ,ₜ','Preactivation gradient','Loss derivative after passing through a component’s activation.'],
 partial:['∂','Partial derivative','Sensitivity to one variable while differentiating the computational graph.'],
 theta:['θ','Parameters','The complete collection of trainable weights and biases.'],
 G:['G','Batch gradient','Average parameter gradient across the healthy and warning examples.'],
 eta:['η','Learning rate','Multiplier controlling the size of a gradient-descent update.'],
 C:['C','Clipping limit','Maximum gradient-vector norm before rescaling; zero disables clipping here.'],
 alpha:['α','Clipping scale','Rescales the whole batch gradient without changing its direction.'],
 D:['D','Input dimension','Number of input features; three in this task.'],
 H:['H','Hidden size','Number of hidden units and entries in each state vector.'],
 T:['T','Final time step','Sequence length; 32 in the worked example.'],
 t:['t','Time index','Position of the current observation in the sequence.'],
 tau:['τ','Decision threshold','Task policy for deciding check versus no check; not part of the loss.'],
 sum:['Σ','Summation','Add contributions, such as all uses of a shared parameter.'],
 transpose:['ᵀ','Transpose','Swap matrix axes; converts vectors into rows for outer products.'],
 nabla:['∇','Gradient operator','Collect derivatives with respect to every entry of a parameter vector or matrix.'],
 a:['a','Component index','Identifies the forget, input, candidate, or output component.'],
 j:['j','Entry index','Indexes individual parameter-vector entries in the gradient norm.'],
 k:['k','Time index in a product','A running index over time steps in the direct memory-path product.']
};
const macros={
 '\\sym':'\\htmlData{notation=#1}{#2}',
 '\\xt':'\\sym{x}{x_t}','\\ht':'\\sym{h}{h_t}','\\hp':'\\sym{hp}{h_{t-1}}',
 '\\ct':'\\sym{c}{c_t}','\\cp':'\\sym{cp}{c_{t-1}}',
 '\\gatef':'\\sym{f}{f_t}','\\gatei':'\\sym{i}{i_t}','\\cand':'\\sym{g}{g_t}','\\gateo':'\\sym{o}{o_t}',
 '\\W':'\\sym{W}{W_{#1}}','\\U':'\\sym{U}{U_{#1}}','\\bias':'\\sym{b}{b_{#1}}',
 '\\za':'\\sym{z}{z_{a,t}}','\\sig':'\\sym{sigma}{\\sigma}','\\th':'\\sym{tanh}{\\tanh}',
 '\\had':'\\sym{had}{\\odot}','\\read':'\\sym{r}{r}','\\v':'\\sym{v}{v}','\\by':'\\sym{by}{b_y}',
 '\\prob':'\\sym{p}{p}','\\target':'\\sym{y}{y}','\\loss':'\\sym{L}{L}',
 '\\dh':'\\sym{dh}{\\bar h_t}','\\dhprev':'\\sym{dh}{\\bar h_{t-1}}',
 '\\Dtot':'\\sym{dc}{D_t}','\\dcfuture':'\\sym{dcfuture}{\\bar c_t^{\\mathrm{future}}}',
 '\\dcprev':'\\sym{dcprev}{\\bar c_{t-1}}','\\del':'\\sym{delta}{\\delta_{#1,t}}',
 '\\params':'\\sym{theta}{\\theta}','\\grad':'\\sym{G}{G}',
 '\\clip':'\\sym{C}{C}'
};
const n=v=>Math.abs(v)>0&&(Math.abs(v)<.00001||Math.abs(v)>=10000)?v.toExponential(6):v.toFixed(6);
const texN=v=>n(v).replace(/e([+-]?\d+)$/,'\\times10^{$1}');
function renderMath(node,tex,display=true){
 try{katex.render(tex,node,{displayMode:display,throwOnError:true,macros,trust:context=>context.command==='\\htmlData',strict:'ignore'});}
 catch(error){node.classList.add('parse-error');node.textContent=tex;node.dataset.parseError=error.message;}
}
function equation(parent,tex,label=''){
 if(label){const title=document.createElement('div');title.className='formula-label';title.textContent=label;parent.append(title);}
 const div=document.createElement('div');div.className='tex display';parent.append(div);renderMath(div,tex);
}
function table(parent,headers,rows,id=''){
 const wrap=document.createElement('div');wrap.className='table-wrap';
 const t=document.createElement('table');t.className='numeric-table';if(id)t.id=id;
 const head=document.createElement('thead'),tr=document.createElement('tr');
 for(const h of headers){const th=document.createElement('th');if(h&&h.html)th.innerHTML=h.html;else th.textContent=h;th.scope='col';tr.append(th);}head.append(tr);t.append(head);
 const body=document.createElement('tbody');for(const values of rows){const row=document.createElement('tr');for(const value of values){const cell=document.createElement('td');if(value&&value.html)cell.innerHTML=value.html;else cell.textContent=String(value);row.append(cell);}body.append(row);}t.append(body);wrap.append(t);parent.append(wrap);return t;
}
function paragraph(parent,text,cls=''){const p=document.createElement('p');p.textContent=text;if(cls)p.className=cls;parent.append(p);}
function sym(key,text){return {html:'<span data-notation="'+key+'">'+text+'</span>'};}
function parameterName(item){
 const map={Wf:'W<sub>f</sub>',Uf:'U<sub>f</sub>',bf:'b<sub>f</sub>',Wi:'W<sub>i</sub>',Ui:'U<sub>i</sub>',bi:'b<sub>i</sub>',Wg:'W<sub>g</sub>',Ug:'U<sub>g</sub>',bg:'b<sub>g</sub>',Wo:'W<sub>o</sub>',Uo:'U<sub>o</sub>',bo:'b<sub>o</sub>',v:'v',bv:'b<sub>y</sub>'};
 const key=item.key==='bv'?'by':item.key==='v'?'v':item.key.startsWith('W')?'W':item.key.startsWith('U')?'U':'b';
 return sym(key,map[item.key]+(item.indices.length?'['+item.indices.join(',')+']':''));
}
const staticMath=Array.from(document.querySelectorAll('article .tex'));
staticMath.forEach(node=>renderMath(node,node.textContent));
const glossaryNode=$('glossary');
for(const [key,[symbol,name,brief]]of Object.entries(glossary)){
 const entry=document.createElement('div');entry.className='gloss-item';
 const mark=document.createElement('span');mark.className='gloss-symbol';mark.dataset.notation=key;mark.textContent=symbol;
 const title=document.createElement('strong');title.textContent=name;
 const p=document.createElement('p');p.textContent=brief;entry.append(mark,title,p);glossaryNode.append(entry);
}
const warningTrace=beforeBatch.traces[1],healthyTrace=beforeBatch.traces[0],warningBack=beforeBatch.backs[1],healthyBack=beforeBatch.backs[0];
for(const [label,trace]of [['Warning sequence',warningTrace],['Healthy control',healthyTrace]]){
 const heading=document.createElement('h3');heading.textContent=label;$('forward-tables').append(heading);
 table($('forward-tables'),['t','Event','f','i','g','o','cprev','c','h'],trace.cache.map(d=>[d.t+1,d.t===31?'Inspect':d.t===0&&trace.y===1?'Warning':'Normal',...['f','i','g','o','cp','c','h'].map(key=>n(d[key][0]))]),trace.y?'warning-forward':'healthy-forward');
}
function lossRows(batch){return batch.traces.map((t,i)=>[i?'Warning':'Healthy',t.y,n(t.logit),n(t.prob),n(t.loss)]);}
table($('initial-losses'),['Example',sym('y','Target y'),sym('r','Logit r'),sym('p','Probability p'),sym('L','Loss L')],lossRows(beforeBatch));
equation($('initial-losses'),'L_{\\mathrm{batch}}=\\frac{'+texN(healthyTrace.loss)+'+'+texN(warningTrace.loss)+'}{2}='+texN(beforeBatch.loss));
table($('readout-back-numbers'),['Example','p − y','∂L/∂hT'],beforeBatch.traces.map((t,i)=>[i?'Warning':'Healthy',n(beforeBatch.backs[i].dzOut),n(beforeBatch.backs[i].steps[31].dh[0])]));
const last=warningTrace.cache[31],q=warningBack.steps[31],d1=warningTrace.cache[0],q1=warningBack.steps[0],terminal=$('terminal-back');
equation(terminal,'\\sym{dh}{\\bar h_{32}}=('+texN(warningTrace.prob)+'-1)(8)='+texN(q.dh[0]),'A · Readout correction');
equation(terminal,'\\sym{dc}{D_{32}}=0+('+texN(q.dh[0])+')('+texN(last.o[0])+')(1-\\tanh^2('+texN(last.c[0])+'))='+texN(q.dc[0]),'B · Total memory correction');
equation(terminal,'\\delta_{o,32}=('+texN(q.dh[0])+')\\tanh('+texN(last.c[0])+')('+texN(last.o[0])+')(1-'+texN(last.o[0])+')='+texN(q.dz.o[0]),'C · Output gate');
equation(terminal,'\\delta_{f,32}=('+texN(q.dc[0])+')('+texN(last.cp[0])+')('+texN(last.f[0])+')(1-'+texN(last.f[0])+')='+texN(q.dz.f[0]),'D · Forget gate');
equation(terminal,'\\delta_{i,32}=('+texN(q.dc[0])+')(0)('+texN(last.i[0])+')(1-'+texN(last.i[0])+')=0','E · Input gate: candidate is zero here');
equation(terminal,'\\delta_{g,32}=('+texN(q.dc[0])+')('+texN(last.i[0])+')(1-0^2)='+texN(q.dz.g[0]),'F · Candidate: zero content can have a nonzero derivative');
equation(terminal,'\\sym{dcprev}{\\bar c_{31}}=('+texN(q.dc[0])+')('+texN(last.f[0])+')='+texN(q.dcp[0])+',\\qquad \\sym{dh}{\\bar h_{31}}=0','G · Propagate backward; all initial U values are zero');
const first=$('first-back');
equation(first,'\\sym{dc}{D_1}=\\sym{dc}{D_{32}}\\prod_{k=2}^{32}f_k='+texN(q1.dc[0]),'Direct memory correction reaches the early warning');
equation(first,'\\delta_{g,1}=('+texN(q1.dc[0])+')('+texN(d1.i[0])+')(1-'+texN(d1.g[0])+'^2)='+texN(q1.dz.g[0]));
equation(first,'\\delta_{i,1}='+texN(q1.dz.i[0])+',\\quad\\delta_{f,1}=0,\\quad\\delta_{o,1}=0');
paragraph(first,'The first forget gradient is zero because c₀ = 0. The first output gradient is zero in this initialization because only the final hidden state is supervised and the recurrent U values are all zero. These are local structural zeros, not evidence that the cell-state correction vanished.');
for(const [label,trace,back]of [['Warning sequence',warningTrace,warningBack],['Healthy control',healthyTrace,healthyBack]]){
 const heading=document.createElement('h3');heading.textContent=label+' · reverse order';$('backward-tables').append(heading);
 table($('backward-tables'),['t','dh','D = total dc','δf','δi','δg','δo','dcprev','dhprev'],back.steps.slice().reverse().map(d=>[d.t+1,n(d.dh[0]),n(d.dc[0]),n(d.dz.f[0]),n(d.dz.i[0]),n(d.dz.g[0]),n(d.dz.o[0]),n(d.dcp[0]),n(d.dhp[0])]),trace.y?'warning-backward':'healthy-backward');
}
table($('epoch-before'),['Example','Target','Old probability','Old loss'],beforeBatch.traces.map((t,i)=>[i?'Warning':'Healthy',t.y,n(t.prob),n(t.loss)]));
paragraph($('epoch-before'),'Mean old loss: '+n(beforeBatch.loss),'epoch-result');
equation($('epoch-clip'),'\\lVert G\\rVert_2='+texN(beforeBatch.norm)+'<C=1,\\qquad\\alpha=1');
paragraph($('epoch-clip'),'The first LSTM batch needs no clipping. For comparison, the initial RNN batch has norm '+n(M.batch(M.create('RNN',cfg),cfg).norm)+'; the same limit rescales that RNN vector. The SGD and clipping equations are identical for both architectures.');
const updates=M.entries(before).map(item=>{
 const healthy=M.valueAt(healthyBack.grad,item),warn=M.valueAt(warningBack.grad,item),avg=M.valueAt(beforeBatch.grad,item),newValue=M.valueAt(model.p,item);
 return [parameterName(item),n(item.value),n(healthy),n(warn),n(avg),n(result.update.factor*avg),n(newValue-item.value),n(newValue)];
});
table($('update-table'),['Parameter','Old θ','Healthy ∂L/∂θ','Warning ∂L/∂θ','Mean Gθ','Clipped Gθ','Change Δθ','New θ'],updates,'all-updates');
const work=$('worked-update'),g=beforeBatch.grad.Wg[0][0],old=before.Wg[0][0],newWeight=model.p.Wg[0][0];
equation(work,'\\frac{\\partial L_{\\mathrm{warning}}}{\\partial W_{g,0,0}}=\\sum_t\\delta_{g,t}x_{t,0}='+texN(q1.dz.g[0])+'\\cdot1='+texN(warningBack.grad.Wg[0][0]));
paragraph(work,'Only the warning token has input feature 0 equal to one. The other 31 steps contribute zero to this particular input weight. The healthy sequence contributes zero because that feature is never active.');
equation(work,'G_{W_{g,0,0}}=\\frac{0+('+texN(warningBack.grad.Wg[0][0])+')}{2}='+texN(g));
equation(work,'W_{g,0,0}^{\\mathrm{new}}='+texN(old)+'-('+cfg.rate+')(1)('+texN(g)+')='+texN(newWeight));
paragraph(work,'The negative gradient increases the weight slightly. This strengthens the proposed representation of a warning. It is one member of the simultaneous 22-parameter update, not a separate training event.');
table($('epoch-after'),['Example','Old probability','New probability','Old loss','New loss'],beforeBatch.traces.map((t,i)=>[i?'Warning':'Healthy',n(t.prob),n(afterBatch.traces[i].prob),n(t.loss),n(afterBatch.traces[i].loss)]),'after-epoch');
paragraph($('epoch-after'),'Mean loss: '+n(beforeBatch.loss)+' → '+n(afterBatch.loss)+'. One full epoch and one batch SGD update are now complete.','epoch-result');
const automatic={'x':'x','h':'h','c':'c','f':'f','i':'i','g':'g','o':'o','W':'W','U':'U','b':'b','z':'z','r':'r','v':'v','p':'p','y':'y','L':'L','G':'G','D':'D','H':'H','T':'T','t':'t','η':'eta','α':'alpha','τ':'tau','θ':'theta','δ':'delta','∂':'partial','σ':'sigma','tanh':'tanh','⊙':'had','∑':'sum','⊤':'transpose','∇':'nabla','a':'a','j':'j','k':'k'};
for(const span of document.querySelectorAll('.katex-html .mathnormal,.katex-html .mord,.katex-html .mop,.katex-html .mbin')){
 const key=automatic[span.textContent];if(key&&!span.closest('[data-notation]'))span.dataset.notation=key;
}
const tip=$('notation-tooltip');let owner=null;
function hide(){tip.hidden=true;if(owner)owner.removeAttribute('aria-describedby');owner=null;}
function show(node){const entry=glossary[node.dataset.notation];if(!entry)return;if(owner&&owner!==node)owner.removeAttribute('aria-describedby');owner=node;tip.replaceChildren();const strong=document.createElement('strong');strong.textContent=entry[0]+' · '+entry[1];const brief=document.createElement('span');brief.textContent=entry[2];tip.append(strong,brief);tip.hidden=false;node.setAttribute('aria-describedby',tip.id);
 const r=node.getBoundingClientRect(),w=tip.offsetWidth,h=tip.offsetHeight;const left=Math.max(12,Math.min(innerWidth-w-12,r.left+r.width/2-w/2));let top=r.top-h-10;if(top<12)top=r.bottom+10;top=Math.max(12,Math.min(innerHeight-h-12,top));tip.style.left=left+'px';tip.style.top=top+'px';
}
for(const node of document.querySelectorAll('[data-notation]')){if(node.closest('.katex-mathml'))continue;const entry=glossary[node.dataset.notation];if(!entry)continue;node.classList.add('notation');node.tabIndex=0;node.setAttribute('role','button');node.setAttribute('aria-label',entry[0]+': '+entry[1]+'. '+entry[2]);node.addEventListener('pointerenter',event=>{if(event.pointerType!=='touch')show(node);});node.addEventListener('pointerleave',()=>{if(document.activeElement!==node)hide();});node.addEventListener('focus',()=>show(node));node.addEventListener('blur',hide);node.addEventListener('click',event=>{event.stopPropagation();show(node);});node.addEventListener('keydown',event=>{if(event.key==='Enter'||event.code==='Space'){event.preventDefault();show(node);}if(event.key==='Escape')hide();});}
document.addEventListener('click',event=>{if(!event.target.closest('[data-notation]'))hide();});window.addEventListener('scroll',hide,{passive:true});window.addEventListener('resize',hide);document.addEventListener('keydown',event=>{if(event.key==='Escape')hide();});
const chapterLinks=Array.from(document.querySelectorAll('.contents a'));
const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting)chapterLinks.forEach(link=>link.classList.toggle('active',link.hash==='#'+entry.target.id));},{rootMargin:'-90px 0px -65% 0px'});
document.querySelectorAll('article section').forEach(section=>observer.observe(section));
})();

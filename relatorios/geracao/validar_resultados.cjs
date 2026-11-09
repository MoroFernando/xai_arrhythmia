// Validador complementar: somente leitura/agregacao; escreve apenas seu parecer.
'use strict';
const a=require('./artefatos.cjs');
const {read,json,hash,write,walk,npz,metrics,mean,same,P,loadRuns}=a;
const checks=[];
function check(nome,ok){checks.push({verificacao:nome,ok:Boolean(ok)});}
function compare(name,m,stored){for(const k of ['n','tn','fp','fn','tp','acuracia','acuracia_balanceada','precisao','sensibilidade','especificidade','f1','auc_roc'])check(name+': '+k,Math.abs(m[k]-stored[k])<=1e-10);}
const manifest=json('relatorios/geracao/manifesto_artefatos_originais.json');
for(const [p,h]of Object.entries(manifest.arquivos))check('Original preservado: '+p,hash(p)===h);
const runs=loadRuns(),comparison=json(P+'outputs/comparacao_politicas/comparacao_sem_vs_com.json');
for(const r of runs){
 const pred=r.p,cam=npz(r.dir+'gradcam_ludb/ludb_gradcam.npz',['checkpoint_sha256','predictions_sha256','dataset_sha256','beat_ids','y_true','y_pred','y_score','checkpoint_role','checkpoint_epoch']),fid=r.iou;
 const best=hash(r.dir+'model.pt'),last=hash(r.dir+'model_last.pt');
 check(r.name+': hash BEST',best===r.m.checkpoints['model.pt'].sha256);
 check(r.name+': hash LAST',last===r.m.checkpoints['model_last.pt'].sha256);
 for(const [name,data]of [['predicoes',pred],['Grad-CAM',cam],['IoU',fid]]){
  check(r.name+': '+name+' BEST hash',data.checkpoint_sha256[0]===best&&data.checkpoint_sha256[0]!==last);
  check(r.name+': '+name+' papel BEST',data.checkpoint_role[0]==='best');
  check(r.name+': '+name+' epoca',data.checkpoint_epoch[0]===r.m.checkpoint_epoch);
  check(r.name+': '+name+' dataset',data.dataset_sha256[0]===hash(P+'data/processed/ludb_'+r.slug+'.npz'));
  check(r.name+': '+name+' identidade',same(data.beat_ids,pred.beat_ids)&&same(data.y_true,pred.y_true));
 }
 check(r.name+': Grad-CAM vinculado a predicoes',cam.predictions_sha256[0]===hash(r.dir+'teste_ludb/ludb_predictions.npz'));
 check(r.name+': IoU vinculado a mapas',fid.gradcam_sha256[0]===hash(r.dir+'gradcam_ludb/ludb_gradcam.npz'));
 check(r.name+': classes previstas iguais',same(cam.y_pred,pred.y_pred)&&same(fid.y_pred,pred.y_pred));
 check(r.name+': diferenca numerica de scores tolerada',Math.max(...cam.y_score.map((v,j)=>Math.abs(v-pred.y_score[j])))<=1e-4);
 check(r.name+': predicao corresponde a threshold 0,5',same(pred.y_pred,pred.y_score.map(v=>Number(v>=.5))));
 const head=r.history[0],rows=r.history.slice(1).map(row=>Object.fromEntries(row.map((v,j)=>[head[j],v]))),bestRow=rows.reduce((b,v)=>Number(v.val_acuracia_balanceada)>Number(b.val_acuracia_balanceada)?v:b),sel=r.m.metrics.selecao_checkpoint;
 check(r.name+': BEST maximiza todo o historico',Number(bestRow.epoca)===sel.best_epoch);
 check(r.name+': BA BEST corresponde ao CSV',Math.abs(Number(bestRow.val_acuracia_balanceada)-sel.best_validation_balanced_accuracy)<=5.1e-7);
 check(r.name+': LAST corresponde ao CSV',Number(rows.at(-1).epoca)===sel.last_epoch&&Math.abs(Number(rows.at(-1).val_acuracia_balanceada)-sel.last_validation_balanced_accuracy)<=5.1e-7);
 compare(r.name+' LUDB independente',metrics(pred.y_true,pred.y_score),r.t.metrics);
 const mt=npz(P+'data/processed/mitbih_'+r.slug+'.npz',['labels','simbolos','pacientes','subject_ids']),man=r.m.split_manifest;
 check(r.name+': MIT dataset vinculado',hash(P+r.m.dataset_path)===r.m.dataset_sha256);
 check(r.name+': MIT 201 e 202 mesmo individuo',mt.pacientes.includes('201')&&mt.pacientes.includes('202')&&mt.subject_ids.every((v,j)=>!['201','202'].includes(mt.pacientes[j])||v==='201'));
 check(r.name+': manifesto 201/202 validacao',man.record_subject['201']==='201'&&man.record_subject['202']==='201'&&man.subject_partition['201']==='validation');
 check(r.name+': sujeitos disjuntos',!r.m.split.treino.sujeitos.some(s=>r.m.split.validacao.sujeitos.includes(s)));
 check(r.name+': pacing MIT excluido',!mt.pacientes.some(s=>['102','104','107','217'].includes(s))&&!mt.simbolos.some(s=>['/','f'].includes(s)));
 const cr=Object.values(r.t.dataset_metadata.clinical_by_record);
 check(r.name+': Ischemia nao gera zero',!cr.some(c=>c.categories.ischemia&&c.label===0));
 check(r.name+': pacing LUDB excluido',!cr.some(c=>c.categories.cardiac_pacing&&c.included_beats>0));
 check(r.name+': Rhythms declarado fora do label',r.t.ludb_negative_reference.rhythms_affect_label===false);
 check(r.name+': positivos possuem alvo',!cr.some(c=>c.label===1&&!c.target_families.length));
 for(const [key,stored]of [['iou_por_batimento',r.f.geral],['iou_p',r.f.por_estrutura['P (mascara_p)']],['iou_qrs',r.f.por_estrutura['QRS (mascara_qrs)']],['iou_t',r.f.por_estrutura['T (mascara_t)']]])check(r.name+': media do IoU salvo '+key,Math.abs(mean(fid[key])-stored.iou_medio)<=1e-7);
}
const [sem,com]=runs,ps=sem.p,pc=com.p,idx=new Map(pc.beat_ids.map((id,j)=>[id,j])),pairs=ps.beat_ids.map((id,j)=>[j,idx.get(id)]).filter(v=>v[1]!==undefined);
check('Intersecao possui 192 beats',pairs.length===192);
check('Intersecao cobre toda populacao SEM',pairs.length===ps.beat_ids.length);
check('Intersecao tem y_true iguais',pairs.every(([s,c])=>ps.y_true[s]===pc.y_true[c]));
check('Intersecao tem registros iguais',pairs.every(([s,c])=>ps.pacientes[s]===pc.pacientes[c]));
const ids=new Set(ps.beat_ids),shared=pc.beat_ids.map(b=>ids.has(b));
check('Shared subset COM = intersecao por identidade',same(shared.map(Number),pc.is_shared_subset));
compare('COM comum independente',metrics(pairs.map(([s,c])=>pc.y_true[c]),pairs.map(([s,c])=>pc.y_score[c])),comparison.mesmos_beats_sem_vs_com.metrics.COM);
const tr={'0->0':0,'0->1':0,'1->0':0,'1->1':0};for(const [s,c]of pairs)tr[ps.y_pred[s]+'->'+pc.y_pred[c]]++;
check('Transicoes conferem com resultado original',same(tr,comparison.mesmos_beats_sem_vs_com.transicoes_pred_SEM_para_COM.todos));
const fam=pc.target_families.map(JSON.parse),sub=pc.target_subtypes.map(JSON.parse),mask=pc.y_true.map((y,j)=>y===0||fam[j].some(f=>['atrial_extrasystole','ventricular_extrasystole'].includes(f))||sub[j].some(t=>['Complete left bundle branch block','Complete right bundle branch block'].includes(t)));
check('Somente completos: mascara recalculada dos alvos',same(mask.map(Number),pc.is_complete_only_eval));
check('Somente completos: n=226',mask.filter(Boolean).length===226);
check('Somente completos: preserva shared subset',mask.every((v,j)=>!shared[j]||v));
compare('COM completos independente',metrics(pc.y_true.filter((v,j)=>mask[j]),pc.y_score.filter((v,j)=>mask[j])),com.t.avaliacao_somente_completos.metrics);
for(const file of walk().filter(p=>p.endsWith('.json'))){try{json(file);check('JSON estrito: '+file,true);}catch{check('JSON estrito: '+file,false);}}
for(const r of runs)for(const file of walk(P+'outputs/notebooks_executados/'+r.slug).filter(p=>p.endsWith('.ipynb'))){const n=json(file),source=n.cells.map(c=>c.source.join('')).join('\n');check('Notebook sem erro salvo: '+file,!n.cells.some(c=>(c.outputs||[]).some(o=>o.output_type==='error')));check('Notebook identificado ATUAL: '+file,source.includes('STATUS: ATUAL'));check('Notebook sem populacao historica no texto: '+file,!/1228|100695/.test(source));if(file.endsWith('06_fidelidade_iou.ipynb')){check(r.name+': NB06 sem recalculo de IoU',!source.includes('res = fidelity.fidelidade('));check(r.name+': NB06 tabela de classes correta',source.includes('n_classe_0')&&source.includes('n_classe_1')&&!source.includes('int(y_true[m][0])'));}}
const docs=["README.md","relatorios/RELATORIO_GERAL_PROJETO.md","relatorios/RELATORIO_VISUAL_RESULTADOS.md","relatorios/GUIA_NOTEBOOKS_RESULTADOS.md","relatorios/RELATORIO_CORRECOES_FINAIS.md"];
const path=require('path'),fs=require('fs');
for(const doc of docs){const content=read(doc).toString('utf8');for(const match of content.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)){const href=match[1];if(/^[a-z]+:|^#/.test(href)||href==='relatorios/geracao/verificacao_revisao_final.json')continue; /* o proprio parecer e gravado ao final */check('Link relativo valido: '+doc+' -> '+href,fs.existsSync(a.local(path.join(path.dirname(doc),href.split('#')[0]))));}}
const failing=checks.filter(c=>!c.ok),report={escopo:'Somente leitura/agregacao de artefatos existentes em pos3; sem pipeline.',data:'2026-10-30',checagens:checks.length,aprovadas:checks.length-failing.length,falhas:failing,verificacoes:checks};
if(!process.argv.includes('--read-only'))write('relatorios/geracao/verificacao_revisao_final.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({checagens:report.checagens,aprovadas:report.aprovadas,falhas:failing},null,2));
if(failing.length)process.exitCode=1;

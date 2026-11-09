'use strict';
const a=require('./artefatos.cjs');
const {read,json,write,walk,loadRuns,esc}=a;
function edit(p,fn){write(p,fn(read(p).toString('utf8')));}
const warning='A cobertura média das regiões clínicas é aproximadamente 0,576. Esse valor não foi validado como baseline aleatório de IoU e não permite afirmar desempenho acima ou abaixo do acaso.';
// Os nomes tecnicos sao mantidos para compatibilidade binaria/JSON.
edit('src/fidelity_ludb.py',s=>s.replace(/\r\n/g,'\n')
 .replace('verdadeira e por estrutura), com a cobertura média da máscara como referência\nao acaso.','verdadeira e por estrutura), com a cobertura média da máscara como estatística\ndescritiva, sem interpretação de baseline aleatório.')
 .replace(/# Referência ao acaso:[\s\S]*?# Ressalva herdada/, '# Compatibilidade: cobertura_acaso e acima_do_acaso sao campos legados/deprecated.\n# O primeiro e cobertura da mascara; o segundo e apenas IoU menos cobertura.\n# Nenhum foi validado como baseline aleatorio ou teste estatistico.\n# Os nomes e valores sao preservados para ler artefatos existentes.\n#\n# Ressalva herdada')
 .replace('cobertura_acaso: fração média da janela coberta pela máscara — a\n            pontuação esperada de um mapa aleatório. Repassada como\n            ``cobertura_acaso`` e usada para ``acima_do_acaso``.', 'cobertura_acaso: argumento legado/deprecated: cobertura média da máscara,\n            sem baseline aleatório validado. Mantido por compatibilidade.\n            ``acima_do_acaso`` é somente a diferença aritmética IoU - cobertura.')
 .replace('classe prevista/verdadeira e a comparação com o acaso.','classe prevista/verdadeira e a cobertura descritiva (campos legados).')
 .replace('f"  esperado ao acaso       : {_fmt(g[\'cobertura_acaso\'])}"','f"  cobertura clinica       : {_fmt(g[\'cobertura_acaso\'])} (descritiva; campo legado)"')
 .replace('f"  acima do acaso          : {_fmt(g[\'acima_do_acaso\'])}"','"  baseline aleatorio IoU  : nao validado; nao interpretar a diferenca legada"'));
for(const p of ['FUNDAMENTACAO.md','ANALISE_CLASSIFICACAO_LUDB.md'])edit(p,s=>s.startsWith('> STATUS: HISTÓRICO')?s:
 '> STATUS: HISTÓRICO — não representa a política utilizada na execução final.\n> Preserve este texto como registro das decisões anteriores. Para a metodologia atual, consulte [o relatório geral](./RELATORIO_GERAL_PROJETO.md) e [as políticas oficiais](./CLASSIFICACAO_ATUALIZADA.md). Flags antigas não estão disponíveis na execução final.\n\n'+s);
const runs=loadRuns();
for(const p of walk('notebooks').filter(p=>p.endsWith('.ipynb'))){const n=json(p);if(!n.cells.some(c=>c.source.join('').includes('STATUS: HISTÓRICO')))
 n.cells.unshift({cell_type:'markdown',metadata:{},source:['> STATUS: HISTÓRICO — outputs preservados; não representam os experimentos finais SEM/COM.\n','> Não execute este notebook para reproduzir os resultados oficiais. Consulte [o guia](../relatorios/GUIA_NOTEBOOKS_RESULTADOS.md).\n','> Referências antigas a cobertura como acaso não são válidas como baseline aleatório de IoU.\n']});
 write(p,JSON.stringify(n,null,1)+'\n');}
write('notebooks/README.md','# Notebooks históricos\n\nOs seis notebooks desta pasta preservam outputs anteriores à execução final, mesmo quando partes do código foram atualizadas. Não são os resultados oficiais. Não foram apagados nem reexecutados.\n\nUse os 12 notebooks de [outputs/notebooks_executados](../outputs/notebooks_executados/) e o [guia completo](../relatorios/GUIA_NOTEBOOKS_RESULTADOS.md).\n');
for(const run of runs){
 const folder='outputs/notebooks_executados/'+run.slug+'/';
 for(const p of walk(folder).filter(p=>p.endsWith('.ipynb'))){const n=json(p);if(!n.cells.some(c=>c.source.join('').includes('STATUS: ATUAL')))
  n.cells.unshift({cell_type:'markdown',metadata:{},source:[`> STATUS: ATUAL — execução final ${run.name}, BEST época ${run.m.checkpoint_epoch}.\n`,'> Revisão editorial em 30/10/2026: somente apresentação de resultados existentes; este notebook não foi reexecutado nesta revisão.\n','> Consulte o [guia oficial](../../../relatorios/GUIA_NOTEBOOKS_RESULTADOS.md). Os notebooks 03/04 contêm chamadas de inferência; seus outputs estão preservados e não devem ser reexecutados nesta etapa editorial.\n']});
 n.metadata.revisao_editorial={data:'2026-10-30',notebook_reexecutado:false,escopo:'Apresentacao; labels, scores, checkpoints e metricas originais preservados'};
 for(const c of n.cells){if(c.cell_type==='markdown'){
  let s=c.source.join('').replace(/1228/g,String(run.p.y_true.length));
  s=s.replace(/Execute antes,[\s\S]*?```bash[\s\S]*?```/g,'Os artefatos desta execução já estão presentes no pacote. Não é necessário executar o pipeline para consultar estes outputs; veja o guia oficial.');
  s=s.replace(/por exemplo, como a fração da energia do mapa que cai[\s\S]*?máscara clínica, por exemplo\)/g,'pelo IoU entre o mapa binarizado e a máscara clínica (sem medir fração de energia)');
  c.source=s.split(/(?<=\n)/);
 }}
 if(p.endsWith('06_fidelidade_iou.ipynb')){
  for(const c of n.cells){let s=c.source.join('');
   if(c.cell_type==='markdown'&&s.includes('A leitura do número:'))s='A cobertura clínica é uma estatística descritiva de área anotada. '+warning+' Os campos técnicos antigos dos JSONs são deprecated; os valores de IoU permanecem inalterados.';
   s=s.replaceAll('com a referência ao acaso marcada','sem linha de baseline aleatório').replaceAll('cobertura_acaso','cobertura_mascara').replaceAll('cobertura media da mascara clinica (acaso)','cobertura media da mascara clinica (descritiva)');
   s=s.replace(/^.*axvline\(cobertura_mascara.*\n?/gm,'');
   if(c.cell_type==='code'&&s.includes('res = fidelity.fidelidade('))s=`# Leitura dos resultados ja calculados: nao chama fidelity.fidelidade.\nimport json\nres = json.loads((RUN_DIR / "fidelidade_ludb" / "ludb_fidelity.json").read_text(encoding="utf-8"))\nwith np.load(RUN_DIR / "fidelidade_ludb" / "ludb_fidelity.npz", allow_pickle=False) as salvo:\n    iou = salvo["iou_por_batimento"].copy()\nprint(fidelity.formata_fidelidade("LUDB", res))`;
   if(c.cell_type==='code'&&s.includes('"classe": int(y_true[m][0])')){
    s=s.replace('"n": int(m.sum()),\n        "classe": int(y_true[m][0]),','"n_total": int(m.sum()),\n        "n_classe_0": int((y_true[m] == 0).sum()),\n        "n_classe_1": int((y_true[m] == 1).sum()),\n        "percentual_classe_0": float((y_true[m] == 0).mean() * 100),\n        "percentual_classe_1": float((y_true[m] == 1).mean() * 100),').replace('sort_values("n",','sort_values("n_total",');
    const rows=[...new Set(run.p.ritmos)].map(r=>{const ix=run.p.ritmos.map((v,j)=>v===r?j:-1).filter(j=>j>=0),n0=ix.filter(j=>run.p.y_true[j]===0).length;return[r,ix.length,n0,ix.length-n0,100*n0/ix.length,100*(ix.length-n0)/ix.length,a.mean(ix.map(j=>run.iou.iou_por_batimento[j])),ix.filter(j=>run.p.y_true[j]===run.p.y_pred[j]).length/ix.length];}).sort((a,b)=>b[1]-a[1]);
    const heads=['ritmo','n_total','n_classe_0','n_classe_1','percentual_classe_0','percentual_classe_1','iou_medio','acerto'];
    const values=rows.map(r=>r.map((v,j)=>j>=4?Number(v).toFixed(j<6?2:6):String(v)));
    c.outputs=[{output_type:'display_data',metadata:{},data:{'text/plain':[heads.join(' | ')+'\n'+values.map(r=>r.join(' | ')).join('\n')],'text/html':['<table><thead><tr>'+heads.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+values.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>']}}];
   }
   if(c.cell_type==='code'&&s.includes('hist(iou[~np.isnan(iou)]')){const svg=read('relatorios/geracao/figuras/distribuicao_iou_'+run.name.toLowerCase()+'.svg').toString();c.outputs=[{output_type:'display_data',metadata:{},data:{'image/svg+xml':[svg],'text/plain':['Distribuicao dos IoUs salvos; sem baseline aleatorio.']}}];}
   for(const o of c.outputs||[]){if(o.text)o.text=o.text.flatMap(line=>line.includes('acima do acaso')?[]:[line.replace('esperado ao acaso','cobertura clinica').replace('(acaso)','(descritiva)')]);}
   c.source=s.split(/(?<=\n)/);
  }
 }
 write(p,JSON.stringify(n,null,1)+'\n');
 }
}
edit('RELATORIO_EXECUCAO_FINAL.md',s=>{
 s=s.replaceAll('**CORRIGIDO**: duas execuções produziram pesos idênticos','**CONFIGURADO**: seeds e cuDNN; igualdade entre repetições independentes não comprovada no pacote');
 s=s.replaceAll('Treino reprodutível: pesos idênticos em duas execuções.','Seeds configuradas e cuDNN determinístico; não há evidência suficiente no pacote para comprovar pesos idênticos entre repetições independentes.');
 s=s.replaceAll('acaso = cobertura da máscara 0,5763','cobertura descritiva da máscara = 0,5763; sem baseline aleatório validado').replaceAll('(acaso 0,5764)','(cobertura descritiva 0,5764; sem baseline aleatório validado)');
 s=s.replace('Nas duas políticas, o IoU global fica muito abaixo do acaso (≈ 0,576): o mapa binarizado é estreito e concentrado no QRS.','Nas duas políticas, os IoUs globais são 0,1141 e 0,1186. A cobertura clínica é ≈ 0,576; ela não é um baseline aleatório de IoU validado. A maior sobreposição por estrutura ocorre no QRS.');
 s=s.replaceAll('`requirements.txt` aponta para PyTorch cu124, sem suporte a GPUs RTX 50xx (Blackwell).','O ambiente de PyTorch/CUDA precisa ser instalado separadamente dos requisitos gerais.');
 s=s.replaceAll('**PENDENTE** (ambiente usou cu128; arquivo não alterado)','**DOCUMENTADO** em AMBIENTE_REPRODUTIBILIDADE.md; dependências gerais e PyTorch separados');
 s=s.replaceAll('`requirements.txt` instala PyTorch cu124, que não suporta RTX 50xx; quem usar essas GPUs precisa do índice cu128.','Instalação atual documentada em AMBIENTE_REPRODUTIBILIDADE.md: PyTorch 2.11.0+cu128 separado; versões das demais dependências não foram congeladas na execução.');
 return s.startsWith('> REVISÃO EDITORIAL')?s:'> REVISÃO EDITORIAL — 30/10/2026. Resultados numéricos originais preservados. Consulte também os novos relatórios geral/visual e o guia na raiz de pos3. As 85 verificações citadas abaixo pertencem à execução original; a validação complementar está em relatorios/geracao/.\n\n'+s;
});
console.log('Apresentacao e documentos existentes atualizados; artefatos numericos preservados.');

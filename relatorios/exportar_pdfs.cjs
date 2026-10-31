// Documentacao e exportacao local para PDF; nao executa o pipeline.
'use strict';
const a=require('./artefatos.cjs');
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url'),{spawnSync}=require('child_process');
const browser='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const files=a.walk().filter(p=>p.endsWith('.md')&&!p.startsWith('relatorios_pdf/')&&!p.startsWith('relatorios/exportacao_html/'));
const legend=`<!-- legenda-interna-projeto -->
## Legenda de nomes e abreviações do projeto

| Nome / abreviação | Significado nesta documentação |
| --- | --- |
| **SEM** / **SEM_BLOQUEIO** | **Sem bloqueio como alvo de classificação**: modelo treinado para reconhecer as extrasístoles compartilhadas entre MIT e LUDB. Bloqueios não são alvos positivos dessa política. O nome não garante ausência de bloqueios como comorbidade em todo registro. |
| **COM** / **COM_BLOQUEIO** | **Com bloqueio como alvo de classificação**: modelo que inclui extrasístoles e bloqueios de ramo completos e incompletos entre seus alvos positivos. |
| **COM somente completos** | Recorte de avaliação do mesmo modelo COM: exclui apenas positivos cuja única justificativa seja bloqueio incompleto. Mantém extrasístoles associadas a incompletos; não é um terceiro modelo. |
| **BEST** / **LAST** | BEST é o checkpoint da melhor época pela Balanced Accuracy da validação MIT; LAST é o da última época. Checkpoint significa o arquivo com os pesos e metadados do modelo. |
| **BA** | Balanced Accuracy, ou acurácia balanceada: média entre sensibilidade e especificidade. |
| **Política / slug** | Regra de seleção dos alvos e seu identificador nos arquivos: shared-extrasystoles-v1 = SEM; shared-extrasystoles-bbb-v1 = COM. |
| **beat / beat_id / n** | Beat é a janela de um batimento; beat_id é seu identificador; n é a quantidade de exemplos do recorte. As classes 0 e 1 são, respectivamente, referência negativa e alvo positivo da política. |

Esta legenda esclarece a nomenclatura interna atual. Em documentos marcados **HISTÓRICO**, não substitui nem atualiza as regras antigas descritas no corpo. As siglas médicas mantêm a nomenclatura dos dados originais.
<!-- /legenda-interna-projeto -->
`;
for(const p of files){let text=a.read(p).toString('utf8');if(text.includes('<!-- legenda-interna-projeto -->'))continue;const h=/^# [^\r\n]+\r?\n/m.exec(text);if(h){const end=h.index+h[0].length;text=text.slice(0,end)+'\n'+legend+'\n'+text.slice(end);}else text=legend+'\n'+text;a.write(p,text);}
function inline(s){const tokens=[];function token(html){tokens.push(html);return '\u0001'+(tokens.length-1)+'\u0002';}
 s=s.replace(/`([^`]+)`/g,(_,v)=>token('<code>'+a.esc(v)+'</code>'));
 s=s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,(_,alt,url)=>token('<img alt="'+a.esc(alt)+'" src="'+a.esc(url)+'"/>'));
 s=s.replace(/\[([^\]]*)\]\(([^)]+)\)/g,(_,label,url)=>token('<a href="'+a.esc(url)+'">'+a.esc(label)+'</a>'));
 s=a.esc(s).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\*([^*]+)\*/g,'<em>$1</em>');
 return s.replace(/\u0001(\d+)\u0002/g,(_,n)=>tokens[+n]);
}
function render(md){let out=[],inTable=false,tableHead=false,list=null,fence=false,code=[],lang='',quote=false;
 const closeTable=()=>{if(inTable){out.push('</tbody></table>');inTable=false;}};
 const closeList=()=>{if(list){out.push('</'+list+'>');list=null;}};
 const closeQuote=()=>{if(quote){out.push('</blockquote>');quote=false;}};
 for(const line of md.split(/\r?\n/)){
  if(/^\s*```/.test(line)){if(fence){out.push('<pre><code>'+a.esc(code.join('\n'))+'</code></pre>');code=[];fence=false;}else{closeTable();closeList();closeQuote();fence=true;lang=line.slice(3);}continue;}
  if(fence){code.push(line);continue;}if(/^<!--.*-->$/.test(line.trim()))continue;
  if(line.startsWith('|')){closeList();closeQuote();if(!inTable){out.push('<table>');inTable=true;tableHead=true;}if(/^\|\s*:?-{3}/.test(line))continue;let cols=line.trim().replace(/^\||\|$/g,'').split(/(?<!\\)\|/).map(v=>v.trim().replace(/\\\|/g,'|'));
   if(tableHead){out.push('<thead><tr>'+cols.map(v=>'<th>'+inline(v)+'</th>').join('')+'</tr></thead><tbody>');tableHead=false;}else out.push('<tr>'+cols.map(v=>'<td>'+inline(v)+'</td>').join('')+'</tr>');continue;}
  closeTable();
  if(/^\s*>/.test(line)){closeList();if(!quote){out.push('<blockquote>');quote=true;}out.push('<p>'+inline(line.replace(/^\s*>\s?/,''))+'</p>');continue;}closeQuote();
  const li=/^\s*(?:([-*]) |(\d+)\. )(.*)/.exec(line);if(li){const kind=li[2]?'ol':'ul';if(list!==kind){closeList();out.push('<'+kind+'>');list=kind;}out.push('<li>'+inline(li[3])+'</li>');continue;}closeList();
  const h=/^(#{1,6}) (.*)/.exec(line);if(h){out.push('<h'+h[1].length+'>'+inline(h[2])+'</h'+h[1].length+'>');continue;}if(/^---+$/.test(line.trim())){out.push('<hr/>');continue;}if(line.trim())out.push('<p>'+inline(line)+'</p>');
 }
 closeTable();closeList();closeQuote();return out.join('\n');
}
const css=`@page{size:A4 landscape;margin:13mm 12mm}*{box-sizing:border-box}body{font:10pt/1.42 Arial,sans-serif;color:#183342;margin:0}h1{font-size:22pt;margin:0 0 14pt}h2{font-size:16pt;margin:20pt 0 9pt;break-after:avoid}h3{font-size:12pt;margin-top:15pt;break-after:avoid}p{margin:5pt 0 8pt}img{display:block;max-width:100%;max-height:168mm;object-fit:contain;margin:9pt auto;break-inside:avoid}table{border-collapse:collapse;width:100%;margin:9pt 0 14pt;font-size:8pt;table-layout:auto}th,td{border:1px solid #cbd9df;padding:5pt;vertical-align:top;overflow-wrap:anywhere}th{background:#e6f0f4;color:#153c4b}thead{display:table-header-group}tr{break-inside:avoid}a{color:#126e82;text-decoration:none}pre{font:8pt/1.35 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:8pt;border:1px solid #dbe3e7}code{font-size:.92em;overflow-wrap:anywhere}blockquote{border-left:3px solid #d66a32;padding:4pt 10pt;margin:10pt 0;background:#fff7ee}li{margin:3pt 0}hr{border:0;border-top:1px solid #d6e1e6;margin:14pt 0}`;
const htmlPaths=[];
for(const p of files){const md=a.read(p).toString('utf8'),title=md.match(/^# (.*)$/m)?.[1]||path.basename(p);const base=(path.relative(path.dirname(a.local('relatorios/exportacao_html/'+p.replace(/\.md$/,'.html'))),a.local(path.dirname(p))).replaceAll('\\','/')||'.')+'/';
 const html='<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><base href="'+a.esc(base)+'"><title>'+a.esc(title)+'</title><style>'+css+'</style></head><body>'+render(md)+'</body></html>';
 const target='relatorios/exportacao_html/'+p.replace(/\.md$/,'.html');a.write(target,html);htmlPaths.push({md:p,html:target,pdf:'relatorios_pdf/'+p.replace(/\.md$/,'.pdf')});
}
// Mantem a versao HTML de consulta do relatorio visual sincronizada com o MD.
const visual=htmlPaths.find(v=>v.md==='RELATORIO_VISUAL_RESULTADOS.md');a.write('RELATORIO_VISUAL_RESULTADOS.html',a.read(visual.html).toString('utf8').replace(/<base\b[^>]*>/gi,''));
const indexTitle='Índice dos relatórios em PDF';
const indexHtml='<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>'+indexTitle+'</title><style>'+css+'</style></head><body><h1>'+indexTitle+'</h1><p>Versões PDF dos documentos Markdown de pos3. Os arquivos .md permanecem em seus locais originais. COM = com bloqueio como alvo; SEM = sem bloqueio como alvo. Consulte a legenda de cada documento.</p><p>Os caminhos abaixo são relativos à pasta relatorios_pdf. Documentos históricos permanecem identificados no próprio texto.</p><table><thead><tr><th>Arquivo PDF</th><th>Fonte Markdown preservada</th></tr></thead><tbody>'+htmlPaths.map(v=>'<tr><td>'+a.esc(v.pdf.replace(/^relatorios_pdf\//,''))+'</td><td>'+a.esc(v.md)+'</td></tr>').join('')+'</tbody></table></body></html>';
a.write('relatorios/exportacao_html/INDICE_RELATORIOS.html',indexHtml);htmlPaths.unshift({md:null,html:'relatorios/exportacao_html/INDICE_RELATORIOS.html',pdf:'relatorios_pdf/INDICE_RELATORIOS.pdf'});
fs.mkdirSync(a.local('relatorios_pdf'),{recursive:true});
const profile=a.local('relatorios/exportacao_html/perfil_edge');
for(const entry of htmlPaths){fs.mkdirSync(path.dirname(a.local(entry.pdf)),{recursive:true});const args=['--headless','--no-sandbox','--disable-gpu-sandbox','--in-process-gpu','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-extensions','--no-pdf-header-footer','--allow-file-access-from-files','--user-data-dir='+profile,'--print-to-pdf='+a.local(entry.pdf),pathToFileURL(a.local(entry.html)).href];const r=spawnSync(browser,args,{windowsHide:true,timeout:60000,encoding:'utf8',maxBuffer:1024*1024});
 if(r.error||!fs.existsSync(a.local(entry.pdf))||a.read(entry.pdf).subarray(0,5).toString()!=='%PDF-')throw Error('Falha PDF '+entry.pdf+': '+(r.error?.message||r.stderr));
 console.log(entry.pdf+' ('+a.read(entry.pdf).length+' bytes)');
}
a.write('relatorios/exportacao_html/manifesto_pdfs.json',JSON.stringify({documentos_md:files.length,pdfs:htmlPaths.map(e=>({fonte:e.md,arquivo:e.pdf,bytes:a.read(e.pdf).length,sha256:a.hash(e.pdf)}))},null,2)+'\n');
console.log('PDFs criados: '+htmlPaths.length+'. Markdown preservados nos locais originais, com legenda interna.');

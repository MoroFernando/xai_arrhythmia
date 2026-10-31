// Renderizador local simples para o Markdown visual, sem dependencias externas.
'use strict';
const {read,write,esc}=require('./artefatos.cjs');
const md=read('RELATORIO_VISUAL_RESULTADOS.md').toString('utf8');
function inline(s){return esc(s).replace(/!\[([^\]]*)\]\(([^)]+)\)/g,'<img alt="$1" src="$2" loading="lazy"/>').replace(/\[([^\]]*)\]\(([^)]+)\)/g,'<a href="$2">$1</a>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>');}
let out=[],table=false,header=false,list=false;
for(const line of md.split('\n')){
 if(line.startsWith('|')){if(!table){out.push('<div class="table"><table>');table=true;header=true;}if(/^\|\s*---/.test(line))continue;const tag=header?'th':'td';out.push('<tr>'+line.slice(1,-1).split('|').map(v=>'<'+tag+'>'+inline(v.trim())+'</'+tag+'>').join('')+'</tr>');header=false;continue;}
 if(table){out.push('</table></div>');table=false;}
 if(line.startsWith('- ')){if(!list){out.push('<ul>');list=true;}out.push('<li>'+inline(line.slice(2))+'</li>');continue;}if(list){out.push('</ul>');list=false;}
 const h=/^(#{1,6}) (.*)/.exec(line);if(h){out.push('<h'+h[1].length+'>'+inline(h[2])+'</h'+h[1].length+'>');continue;}
 if(line.trim())out.push('<p>'+inline(line)+'</p>');
}
if(table)out.push('</table></div>');if(list)out.push('</ul>');
write('RELATORIO_VISUAL_RESULTADOS.html','<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Relatório visual — ECG</title><style>body{font:16px/1.6 Arial,sans-serif;color:#183342;max-width:1240px;margin:auto;padding:32px;background:#f6f8fa}h1,h2,h3{line-height:1.25}h2{margin-top:48px}img{display:block;width:100%;height:auto;background:white;border:1px solid #dde5e9;border-radius:8px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border:1px solid #dae2e6;padding:9px;text-align:left}th{background:#e7f0f3}.table{overflow-x:auto}a{color:#126e82}code{background:#e9eef1;padding:2px 4px}@media print{body{background:white;padding:0}img{break-inside:avoid}h2,h3{break-after:avoid}}</style><main>'+out.join('\n')+'</main></html>');
console.log('HTML visual criado com links relativos e sem dependencias de rede.');

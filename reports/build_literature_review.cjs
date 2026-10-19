const fs = require('fs');
const path = require('path');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,AlignmentType,Footer,PageNumber,Table,TableRow,TableCell,WidthType,ShadingType,ExternalHyperlink,TabStopType,BorderStyle} = require('docx');
const root = __dirname;
const source = fs.readFileSync(path.join(root,'literature_review.md'),'utf8');
const pagesPath = path.join(root,'literature_review_preview','contents_pages.json');
const pages = fs.existsSync(pagesPath) ? JSON.parse(fs.readFileSync(pagesPath)) : {};
const lines = source.split('\n');
const major = lines.filter(x=>x.startsWith('## ') && !x.includes('A Critical')).map(x=>x.slice(3));
const run = (text, opts={}) => new TextRun({text,...opts});
const p = (text,opts={})=>new Paragraph({children:[run(text)],...opts});
const children = [
 p('LITERATURE SURVEY',{spacing:{before:1350,after:420},alignment:AlignmentType.CENTER}),
 p('Robust Multimodal\u00a0Deepfake Detection',{style:'Title',alignment:AlignmentType.CENTER,spacing:{after:420}}),
 p('A Critical Literature Survey of Visual, Audio, and Audio-Visual Methods',{style:'Subtitle',alignment:AlignmentType.CENTER,spacing:{after:1600}}),
 p('IEEE numbered references',{alignment:AlignmentType.CENTER}),
 p('13 October 2026',{alignment:AlignmentType.CENTER}),
 p('Critical narrative review | 30 selected references',{alignment:AlignmentType.CENTER}),
 p('Contents',{heading:HeadingLevel.HEADING_1,pageBreakBefore:true})
];
for(const title of major) children.push(new Paragraph({children:[run(title),run('\t'+(pages[title]||''))],tabStops:[{type:TabStopType.RIGHT,position:9026,leader:'dot'}],spacing:{after:170}}));
children.push(p('Tables: Dataset roles and evaluation boundaries; Representative methods and their practical boundaries.',{spacing:{before:260},style:'Note'}));
let i=4;
let inRefs=false;
function table(rows){
 const widths=rows[0].length===4?[1700,2250,2370,2706]:[2250,2240,2350,2186];
 return new Table({width:{size:9026,type:WidthType.DXA},columnWidths:widths,rows:rows.map((row,ri)=>new TableRow({tableHeader:ri===0,cantSplit:true,children:row.map((txt,ci)=>new TableCell({width:{size:widths[ci],type:WidthType.DXA},shading:{type:ShadingType.CLEAR,fill:ri===0?'17324D':ri%2?'F1F4F7':'FFFFFF'},margins:{top:100,bottom:100,left:110,right:110},children:[new Paragraph({children:[run(txt,{size:19,bold:ri===0,color:ri===0?'FFFFFF':'17202A'})],spacing:{after:0,line:245},alignment:AlignmentType.LEFT})]}))}))});
}
for(;i<lines.length;i++){
 const line=lines[i].trim(); if(!line)continue;
 if(line.startsWith('Literature review'))continue;
 if(line.startsWith('## ')){
  const title=line.slice(3);inRefs=title==='References';
  children.push(p(title,{heading:HeadingLevel.HEADING_1,pageBreakBefore:title==='Abstract'||inRefs}));continue;
 }
 if(line.startsWith('### ')){children.push(p(line.slice(4),{heading:HeadingLevel.HEADING_2}));continue;}
 if(line.startsWith('|')){
  const rows=[];
  while(i<lines.length&&lines[i].trim().startsWith('|')){
   const cells=lines[i].trim().split('|').slice(1,-1).map(x=>x.trim());
   if(!cells.every(x=>/^[-:]+$/.test(x)))rows.push(cells);i++;
  }i--;children.push(table(rows));children.push(p('',{spacing:{after:60}}));continue;
 }
 if(inRefs&&/^\[\d+\]/.test(line)){
  const parts=line.split('https://');const url=parts.length>1?'https://'+parts[1]:null;
  const rs=[run(parts[0],{size:19})];
  if(url)rs.push(new ExternalHyperlink({link:url,children:[run(url.replace(/([/_-])/g,'$1\u200b'),{style:'Hyperlink',size:18})]}));
  children.push(new Paragraph({children:rs,spacing:{after:145,line:245},indent:{left:430,hanging:430}}));continue;
 }
 const caption=/^Table \d+\./.test(line);
 children.push(p(line,{...(caption?{style:'Caption',keepNext:true}:{}),...(line.startsWith('Keywords:')?{style:'Note'}:{})}));
}
const doc=new Document({creator:'',title:'Robust Multimodal Deepfake Detection: A Critical Literature Survey',description:'A critical narrative review of visual, audio, and multimodal detection with IEEE references.',styles:{default:{document:{run:{font:'Times New Roman',size:22,color:'17202A'},paragraph:{spacing:{after:150,line:280},alignment:AlignmentType.JUSTIFIED}}},paragraphStyles:[
 {id:'Title',name:'Title',basedOn:'Normal',run:{font:'Times New Roman',size:48,bold:true,color:'17324D'},paragraph:{spacing:{line:560}}},
 {id:'Subtitle',name:'Subtitle',basedOn:'Normal',run:{size:28,color:'43566A'},paragraph:{spacing:{line:340}}},
 {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',run:{size:29,bold:true,color:'17324D'},paragraph:{outlineLevel:0,keepNext:true,spacing:{before:320,after:170},alignment:AlignmentType.LEFT}},
 {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',run:{size:24,bold:true,color:'17324D'},paragraph:{outlineLevel:1,keepNext:true,spacing:{before:230,after:120},alignment:AlignmentType.LEFT}},
 {id:'Caption',name:'Caption',basedOn:'Normal',run:{size:20,bold:true},paragraph:{spacing:{before:220,after:130},alignment:AlignmentType.LEFT}},
 {id:'Note',name:'Note',basedOn:'Normal',run:{size:20,italics:true,color:'43566A'},paragraph:{alignment:AlignmentType.LEFT}}
 ]},sections:[{properties:{titlePage:true,page:{size:{width:11906,height:16838},margin:{top:1200,bottom:1200,left:1440,right:1440}}},footers:{default:new Footer({children:[new Paragraph({children:[run('Literature review  |  ',{size:18,color:'667788'}),run('',{children:[PageNumber.CURRENT],size:18}),run(' / ',{size:18}),run('',{children:[PageNumber.TOTAL_PAGES],size:18})],alignment:AlignmentType.RIGHT,border:{top:{color:'CED7DF',style:BorderStyle.SINGLE,size:4}},spacing:{before:100}})]})},children}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(path.join(root,'literature_review.docx'),b);console.log('Created literature_review.docx');});

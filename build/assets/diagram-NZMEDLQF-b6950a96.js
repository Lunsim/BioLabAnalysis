import{p as w}from"./chunk-BAOP5US2-85befb56.js";import{ai as B,V as S,W as z,a6 as F,a7 as P,X as W,Y as T,_ as i,l as v,z as x,d as D,a9 as _,T as A}from"./chunk-SGE5E2BZ-2772fc9c.js";import{s as E}from"./index-e5a91f3e.js";import{p as N}from"./mermaid-parser.core-7e20b3bc.js";import"./isEmpty-506b32a7.js";import"./chunk-NCMFTTUW-c6ef2752.js";import"./chunk-Y27MQZ3U-a2768810.js";import"./_baseUniq-a26c4e3e.js";import"./_basePickBy-717bbfda.js";import"./clone-292b3682.js";import"./chunk-4YFB5VUC-e5d095cd.js";import"./chunk-EQFLFMNE-bfc32f69.js";import"./chunk-BI6EQKOQ-6ee609a9.js";import"./chunk-FF7BQXOH-ed4e9cc3.js";var C={packet:[]},h=structuredClone(C),Y=B.packet,L=i(()=>{const t=x({...Y,...D().packet});return t.showBits&&(t.paddingY+=10),t},"getConfig"),I=i(()=>h.packet,"getPacket"),M=i(t=>{t.length>0&&h.packet.push(t)},"pushWord"),O=i(()=>{_(),h=structuredClone(C)},"clear"),u={pushWord:M,getPacket:I,getConfig:L,clear:O,setAccTitle:S,getAccTitle:z,setDiagramTitle:F,getDiagramTitle:P,getAccDescription:W,setAccDescription:T},X=1e4,G=i(t=>{w(t,u);let e=-1,o=[],n=1;const{bitsPerRow:s}=u.getConfig();for(let{start:a,end:r,label:p}of t.blocks){if(r&&r<a)throw new Error(`Packet block ${a} - ${r} is invalid. End must be greater than start.`);if(a!==e+1)throw new Error(`Packet block ${a} - ${r??a} is not contiguous. It should start from ${e+1}.`);for(e=r??a,v.debug(`Packet block ${a} - ${e} with label ${p}`);o.length<=s+1&&u.getPacket().length<X;){const[m,c]=H({start:a,end:r,label:p},n,s);if(o.push(m),m.end+1===n*s&&(u.pushWord(o),o=[],n++),!c)break;({start:a,end:r,label:p}=c)}}u.pushWord(o)},"populate"),H=i((t,e,o)=>{if(t.end===void 0&&(t.end=t.start),t.start>t.end)throw new Error(`Block start ${t.start} is greater than block end ${t.end}.`);return t.end+1<=e*o?[t,void 0]:[{start:t.start,end:e*o-1,label:t.label},{start:e*o,end:t.end,label:t.label}]},"getNextFittingBlock"),K={parse:i(async t=>{const e=await N("packet",t);v.debug(e),G(e)},"parse")},R=i((t,e,o,n)=>{const s=n.db,a=s.getConfig(),{rowHeight:r,paddingY:p,bitWidth:m,bitsPerRow:c}=a,b=s.getPacket(),l=s.getDiagramTitle(),g=r+p,d=g*(b.length+1)-(l?0:r),k=m*c+2,f=E(e);f.attr("viewbox",`0 0 ${k} ${d}`),A(f,d,k,a.useMaxWidth);for(const[$,y]of b.entries())U(f,y,$,a);f.append("text").text(l).attr("x",k/2).attr("y",d-g/2).attr("dominant-baseline","middle").attr("text-anchor","middle").attr("class","packetTitle")},"draw"),U=i((t,e,o,{rowHeight:n,paddingX:s,paddingY:a,bitWidth:r,bitsPerRow:p,showBits:m})=>{const c=t.append("g"),b=o*(n+a)+a;for(const l of e){const g=l.start%p*r+1,d=(l.end-l.start+1)*r-s;if(c.append("rect").attr("x",g).attr("y",b).attr("width",d).attr("height",n).attr("class","packetBlock"),c.append("text").attr("x",g+d/2).attr("y",b+n/2).attr("class","packetLabel").attr("dominant-baseline","middle").attr("text-anchor","middle").text(l.label),!m)continue;const k=l.end===l.start,f=b-2;c.append("text").attr("x",g+(k?d/2:0)).attr("y",f).attr("class","packetByte start").attr("dominant-baseline","auto").attr("text-anchor",k?"middle":"start").text(l.start),k||c.append("text").attr("x",g+d).attr("y",f).attr("class","packetByte end").attr("dominant-baseline","auto").attr("text-anchor","end").text(l.end)}},"drawWord"),V={draw:R},j={byteFontSize:"10px",startByteColor:"black",endByteColor:"black",labelColor:"black",labelFontSize:"12px",titleColor:"black",titleFontSize:"14px",blockStrokeColor:"black",blockStrokeWidth:"1",blockFillColor:"#efefef"},q=i(({packet:t}={})=>{const e=x(j,t);return`
	.packetByte {
		font-size: ${e.byteFontSize};
	}
	.packetByte.start {
		fill: ${e.startByteColor};
	}
	.packetByte.end {
		fill: ${e.endByteColor};
	}
	.packetLabel {
		fill: ${e.labelColor};
		font-size: ${e.labelFontSize};
	}
	.packetTitle {
		fill: ${e.titleColor};
		font-size: ${e.titleFontSize};
	}
	.packetBlock {
		stroke: ${e.blockStrokeColor};
		stroke-width: ${e.blockStrokeWidth};
		fill: ${e.blockFillColor};
	}
	`},"styles"),pt={parser:K,db:u,renderer:V,styles:q};export{pt as diagram};

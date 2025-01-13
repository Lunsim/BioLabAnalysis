import{p as U}from"./chunk-BAOP5US2-85befb56.js";import{ay as S,az as b,aA as j,ai as X,a6 as Y,a7 as Z,V as q,W as H,Y as J,X as K,_ as u,l as W,a9 as Q,n as tt,z as et,aB as at,T as rt}from"./chunk-SGE5E2BZ-2772fc9c.js";import{s as nt}from"./index-e5a91f3e.js";import{p as it}from"./mermaid-parser.core-7e20b3bc.js";import{d as P}from"./arc-bda54be8.js";import{o as ot}from"./ordinal-ba9b4969.js";import"./isEmpty-506b32a7.js";import"./chunk-NCMFTTUW-c6ef2752.js";import"./chunk-Y27MQZ3U-a2768810.js";import"./_baseUniq-a26c4e3e.js";import"./_basePickBy-717bbfda.js";import"./clone-292b3682.js";import"./chunk-4YFB5VUC-e5d095cd.js";import"./chunk-EQFLFMNE-bfc32f69.js";import"./chunk-BI6EQKOQ-6ee609a9.js";import"./chunk-FF7BQXOH-ed4e9cc3.js";import"./init-77b53fdd.js";function st(t,a){return a<t?-1:a>t?1:a>=t?0:NaN}function lt(t){return t}function ct(){var t=lt,a=st,m=null,s=S(0),g=S(b),x=S(0);function i(e){var r,l=(e=j(e)).length,c,A,h=0,p=new Array(l),n=new Array(l),v=+s.apply(this,arguments),w=Math.min(b,Math.max(-b,g.apply(this,arguments)-v)),f,T=Math.min(Math.abs(w)/l,x.apply(this,arguments)),$=T*(w<0?-1:1),d;for(r=0;r<l;++r)(d=n[p[r]=r]=+t(e[r],r,e))>0&&(h+=d);for(a!=null?p.sort(function(y,C){return a(n[y],n[C])}):m!=null&&p.sort(function(y,C){return m(e[y],e[C])}),r=0,A=h?(w-l*$)/h:0;r<l;++r,v=f)c=p[r],d=n[c],f=v+(d>0?d*A:0)+$,n[c]={data:e[c],index:r,value:d,startAngle:v,endAngle:f,padAngle:T};return n}return i.value=function(e){return arguments.length?(t=typeof e=="function"?e:S(+e),i):t},i.sortValues=function(e){return arguments.length?(a=e,m=null,i):a},i.sort=function(e){return arguments.length?(m=e,a=null,i):m},i.startAngle=function(e){return arguments.length?(s=typeof e=="function"?e:S(+e),i):s},i.endAngle=function(e){return arguments.length?(g=typeof e=="function"?e:S(+e),i):g},i.padAngle=function(e){return arguments.length?(x=typeof e=="function"?e:S(+e),i):x},i}var R=X.pie,F={sections:new Map,showData:!1,config:R},z=F.sections,G=F.showData,pt=structuredClone(R),ut=u(()=>structuredClone(pt),"getConfig"),gt=u(()=>{z=new Map,G=F.showData,Q()},"clear"),dt=u(({label:t,value:a})=>{z.has(t)||(z.set(t,a),W.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),ft=u(()=>z,"getSections"),mt=u(t=>{G=t},"setShowData"),ht=u(()=>G,"getShowData"),I={getConfig:ut,clear:gt,setDiagramTitle:Y,getDiagramTitle:Z,setAccTitle:q,getAccTitle:H,setAccDescription:J,getAccDescription:K,addSection:dt,getSections:ft,setShowData:mt,getShowData:ht},vt=u((t,a)=>{U(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),yt={parse:u(async t=>{const a=await it("pie",t);W.debug(a),vt(a,I)},"parse")},St=u(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),xt=St,At=u(t=>{const a=[...t.entries()].map(s=>({label:s[0],value:s[1]})).sort((s,g)=>g.value-s.value);return ct().value(s=>s.value)(a)},"createPieArcs"),wt=u((t,a,m,s)=>{W.debug(`rendering pie chart
`+t);const g=s.db,x=tt(),i=et(g.getConfig(),x.pie),e=40,r=18,l=4,c=450,A=c,h=nt(a),p=h.append("g");p.attr("transform","translate("+A/2+","+c/2+")");const{themeVariables:n}=x;let[v]=at(n.pieOuterStrokeWidth);v??(v=2);const w=i.textPosition,f=Math.min(A,c)/2-e,T=P().innerRadius(0).outerRadius(f),$=P().innerRadius(f*w).outerRadius(f*w);p.append("circle").attr("cx",0).attr("cy",0).attr("r",f+v/2).attr("class","pieOuterCircle");const d=g.getSections(),y=At(d),C=[n.pie1,n.pie2,n.pie3,n.pie4,n.pie5,n.pie6,n.pie7,n.pie8,n.pie9,n.pie10,n.pie11,n.pie12],D=ot(C);p.selectAll("mySlices").data(y).enter().append("path").attr("d",T).attr("fill",o=>D(o.data.label)).attr("class","pieCircle");let N=0;d.forEach(o=>{N+=o}),p.selectAll("mySlices").data(y).enter().append("text").text(o=>(o.data.value/N*100).toFixed(0)+"%").attr("transform",o=>"translate("+$.centroid(o)+")").style("text-anchor","middle").attr("class","slice"),p.append("text").text(g.getDiagramTitle()).attr("x",0).attr("y",-(c-50)/2).attr("class","pieTitleText");const M=p.selectAll(".legend").data(D.domain()).enter().append("g").attr("class","legend").attr("transform",(o,k)=>{const E=r+l,_=E*D.domain().length/2,B=12*r,V=k*E-_;return"translate("+B+","+V+")"});M.append("rect").attr("width",r).attr("height",r).style("fill",D).style("stroke",D),M.data(y).append("text").attr("x",r+l).attr("y",r-l).text(o=>{const{label:k,value:E}=o.data;return g.getShowData()?`${k} [${E}]`:k});const L=Math.max(...M.selectAll("text").nodes().map(o=>(o==null?void 0:o.getBoundingClientRect().width)??0)),O=A+e+r+l+L;h.attr("viewBox",`0 0 ${O} ${c}`),rt(h,c,O,i.useMaxWidth)},"draw"),Ct={draw:wt},_t={parser:yt,db:I,renderer:Ct,styles:xt};export{_t as diagram};

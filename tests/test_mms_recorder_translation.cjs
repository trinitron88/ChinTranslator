const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path');
const pythonSource=fs.readFileSync(path.join(__dirname,'..','mms_realtime.py'),'utf8');
const source=pythonSource.match(/RECORDER_JS = r"""([\s\S]*?)"""/)[1];
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function harness(){
 const ids=['mms-start','mms-stop','mms-clear','mms-status','mms-transcript','mms-threshold','mms-english','mms-english-status','mms-translate'];
 const el=Object.fromEntries(ids.map(id=>[id,{dataset:{seconds:'3'},value:'',textContent:'',disabled:false}]));el['mms-threshold'].value='-45';
 let processor,translationRequests=[],asr=0,events={};
 const audioNode=()=>({connect(){},disconnect(){},gain:{value:1}});
 class AudioContext{constructor(){this.sampleRate=16000;this.destination={};}resume(){}close(){}createMediaStreamSource(){return audioNode();}createScriptProcessor(){processor=audioNode();return processor;}createGain(){return audioNode();}}
 const context={document:{getElementById:id=>el[id]},window:{AudioContext,addEventListener:(id,cb)=>events[id]=cb},navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop(){}}]})}},requestAnimationFrame:cb=>cb(),AbortController,Float32Array,console,
  fetch:async(url,options)=>{
   if(url==='mms/transcribe'){asr++;return {ok:true,json:async()=>({text:'Chin '+asr,latency:.5,skipped:false})};}
   assert.equal(url,'mms/translate');
   return new Promise(resolve=>translationRequests.push({options,resolve}));
  }};
 vm.runInNewContext('('+source+')()',context);
 return {el,translationRequests,events,async chunk(){processor.onaudioprocess({inputBuffer:{getChannelData:()=>new Float32Array(48000)}});await tick();},resolve(index,text='English'){translationRequests[index].resolve({ok:true,json:async()=>({text,latency:1,truncated:false})});}};
}
(async()=>{
 const h=harness();await h.el['mms-start'].onclick();
 for(let i=0;i<5;i++)await h.chunk();
 assert.equal(h.translationRequests.length,0,'translation must wait for stop');
 const transcript=h.el['mms-transcript'].value;
 h.el['mms-stop'].onclick();assert.equal(h.translationRequests.length,1);
 assert.equal(JSON.parse(h.translationRequests[0].options.body).text,'Chin 2 Chin 3 Chin 4 Chin 5');
 h.resolve(0);await tick();assert.equal(h.el['mms-english'].value,'English');assert.equal(h.el['mms-transcript'].value,transcript);
 h.el['mms-translate'].onclick();assert.equal(h.translationRequests.length,2);
 h.el['mms-clear'].onclick();assert.equal(h.el['mms-english'].value,'');assert.equal(h.el['mms-transcript'].value,'');assert(h.translationRequests[1].options.signal.aborted);
 h.resolve(1,'Stale');await tick();assert.equal(h.el['mms-english'].value,'','clear ignores old translation');
 await h.el['mms-start'].onclick();await h.chunk();h.el['mms-stop'].onclick();assert.equal(h.translationRequests.length,3);
 await h.el['mms-start'].onclick();h.resolve(2,'Prior session');await tick();assert.equal(h.el['mms-english'].value,'','new session ignores old translation');
 await h.chunk();h.el['mms-stop'].onclick();assert.equal(h.translationRequests.length,4);
 h.translationRequests[3].resolve({ok:false,json:async()=>({error:'busy'})});await tick();assert(h.el['mms-english-status'].textContent.includes('Translation failed'));assert(h.el['mms-transcript'].value.includes('Chin 7'));assert.equal(h.el['mms-translate'].disabled,false);
 console.log('Recorder tests passed: wait for Stop, latest excerpt, separate transcript, clear/reset, stale results, retry after failure.');
})().catch(e=>{console.error(e);process.exitCode=1;});

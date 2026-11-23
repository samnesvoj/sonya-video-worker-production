const { chromium } = require('playwright'); const fs=require('fs'), path=require('path');
const [STAGE, OUT] = process.argv.slice(2);
const CH = process.env.HOME + '/Library/Caches/ms-playwright/chromium_headless_shell-1255/chrome-headless-shell-mac-arm64/chrome-headless-shell';
(async()=>{ const b=await chromium.launch({executablePath:CH, proxy:{server:'socks5://127.0.0.1:18080'}});
 const p=await (await b.newContext({viewport:{width:1200,height:630},deviceScaleFactor:1})).newPage();
 await p.route(/^https:\/\/sonya\.group\//, r=>{const u=new URL(r.request().url()); const f=path.join(STAGE,u.pathname); if(fs.existsSync(f)&&!fs.statSync(f).isDirectory()) return r.fulfill({status:200,contentType:f.endsWith('.html')?'text/html; charset=utf-8':'image/webp',body:fs.readFileSync(f)}); r.continue();});
 await p.goto('https://sonya.group/invite/og.html',{waitUntil:'networkidle',timeout:90000}); p.on('requestfailed',r=>console.log('FAIL',r.url().slice(0,90),r.failure()&&r.failure().errorText)); await p.evaluate(()=>document.fonts.ready); await p.waitForTimeout(1500); console.log(await p.evaluate(()=>[...document.fonts].map(f=>f.family+':'+f.status).join(' ')));
 await p.screenshot({path:OUT,type:'jpeg',quality:86}); await b.close();})();

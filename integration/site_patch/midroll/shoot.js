// node shoot.js <stage_dir> <out_prefix> <script.json>   (NODE_PATH=…/direct-upload/pw/node_modules)
// job: {vp:"ph"|"pc", theme, path, hash, login:bool, fake:{"/api/x":[status,body]}, steps:[["eval",js],["wait",ms],["shot",name,full]]}
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const [STAGE, OUT, SCRIPT] = process.argv.slice(2);
const jobs = JSON.parse(fs.readFileSync(SCRIPT, 'utf8'));
const CHROME = process.env.CHROME || fs.readdirSync(process.env.HOME + '/Library/Caches/ms-playwright/chromium_headless_shell-1255').length &&
  process.env.HOME + '/Library/Caches/ms-playwright/chromium_headless_shell-1255/chrome-headless-shell-mac-arm64/chrome-headless-shell';
const captured = [];
(async () => {
  const b = await chromium.launch({ executablePath: CHROME, proxy: { server: 'socks5://127.0.0.1:18080' } });
  for (const job of jobs) {
    const ph = job.vp === 'ph';
    const ctx = await b.newContext(ph ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 3 }
                                      : { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
    if (job.login) {
      const tok = Object.values(JSON.parse(fs.readFileSync('' + (process.env.TOK || '/Users/agent/SONYA/ux-1118/tokens.json') + '')))[0].token;
      await ctx.addCookies([{ name: 'sonya_session', value: tok, domain: 'sonya.group', path: '/', secure: true, httpOnly: true, sameSite: 'Lax' }]);
    }
    await ctx.addInitScript(([t, extra]) => { try { localStorage.setItem('theme', t); localStorage.setItem('sonya_cookie_consent', JSON.stringify({version:1,choice:'necessary',categories:{necessary:true,analytics:false,marketing:false},ts:Date.now(),updatedAt:new Date().toISOString()})); if (extra) eval(extra); } catch (_) {} }, [job.theme, job.init || '']);
    const p = await ctx.newPage();
    await p.route(/^https:\/\/sonya\.group\//, async r => {
      const req = r.request(); const uu = new URL(req.url());
      const fake = (job.fake || {})[uu.pathname];
      if (req.method() !== 'GET') captured.push({ job: job.name, method: req.method(), url: req.url(), body: req.postData() });
      if (fake) return r.fulfill({ status: fake[0], contentType: 'application/json', body: JSON.stringify(fake[1]) });
      let f = path.join(STAGE, uu.pathname.endsWith('/') ? uu.pathname + 'index.html' : uu.pathname);
      if (uu.pathname.startsWith('/api/') || !fs.existsSync(f) || fs.statSync(f).isDirectory()) return r.continue();
      const ct = f.endsWith('.html') ? 'text/html' : f.endsWith('.css') ? 'text/css' : f.endsWith('.js') ? 'application/javascript'
        : f.endsWith('.webp') ? 'image/webp' : f.endsWith('.png') ? 'image/png' : f.endsWith('.jpg') ? 'image/jpeg' : f.endsWith('.mp4') ? 'video/mp4' : 'application/octet-stream';
      r.fulfill({ status: 200, contentType: ct + (ct.startsWith('text') || ct.includes('javascript') ? '; charset=utf-8' : ''), body: fs.readFileSync(f) });
    });
    p.on('pageerror', e => console.log(job.name, 'PAGEERR', e.message.slice(0, 200)));
    p.on('console', m => { if (m.type() === 'error') console.log(job.name, 'CONSOLE', m.text().slice(0, 200)); });
    await p.goto('https://sonya.group' + (job.path || '/') + (job.hash || ''), { waitUntil: 'load', timeout: 90000 }).catch(e => console.log(job.name, 'GOTO', e.message.slice(0, 80)));
    await p.waitForTimeout(job.settle || 3500);
    for (const st of job.steps || []) {
      if (st[0] === 'file') await p.setInputFiles(st[1], st[2]).catch(e => console.log(job.name, 'FILE', e.message.slice(0, 160)));
      else if (st[0] === 'eval') { const v = await p.evaluate(st[1]).catch(e => 'EVALERR ' + e.message.slice(0, 200)); if (v !== undefined && v !== null) console.log(job.name, 'eval:', JSON.stringify(v).slice(0, 3000)); }
      else if (st[0] === 'wait') await p.waitForTimeout(st[1]);
      else if (st[0] === 'click') await p.click(st[1], { timeout: 5000 }).catch(e => console.log(job.name, 'CLICK', e.message.slice(0, 120)));
      else if (st[0] === 'shot') await p.screenshot({ path: `${OUT}_${st[1]}.png`, fullPage: !!st[2] });
    }
    await ctx.close();
  }
  await b.close();
  fs.writeFileSync(OUT + '_requests.json', JSON.stringify(captured, null, 1));
})();

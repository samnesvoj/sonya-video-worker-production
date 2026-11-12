// TeX on stdin → SVG on stdout, offline. Optional backend, see formula.py. Run only on TeX that passed tex.check_tex.
// SONYA_MATHJAX_DIR: directory holding node_modules/mathjax-full. Packages limited to base + ams; `require` is not loaded.
const path = require('path');
const base = path.join(process.env.SONYA_MATHJAX_DIR, 'node_modules/mathjax-full/js');
const load = (p) => require(path.join(base, p));
const { mathjax } = load('mathjax.js');
const { TeX } = load('input/tex.js');
const { SVG } = load('output/svg.js');
const { liteAdaptor } = load('adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = load('handlers/html.js');
load('input/tex/base/BaseConfiguration.js');
load('input/tex/ams/AmsConfiguration.js');

let tex = '';
process.stdin.on('data', (d) => { tex += d; });
process.stdin.on('end', () => {
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const doc = mathjax.document('', {
    InputJax: new TeX({ packages: ['base', 'ams'] }),
    OutputJax: new SVG({ fontCache: 'none' }),
  });
  const node = doc.convert(tex.trim(), { display: true });
  process.stdout.write(adaptor.innerHTML(node));
});

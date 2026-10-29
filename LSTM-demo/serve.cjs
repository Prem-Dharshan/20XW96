const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = fs.realpathSync(path.join(__dirname, 'dist'));
const types = {'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.woff2':'font/woff2','.ttf':'font/ttf','.txt':'text/plain; charset=utf-8'};
http.createServer((req,res)=>{
  if (!['GET','HEAD'].includes(req.method)) {res.writeHead(405);return res.end();}
  try {
    const url = new URL(req.url,'http://localhost');
    const route = decodeURIComponent(url.pathname);
    const file = fs.realpathSync(path.resolve(root, '.' + (route==='/'?'/index.html':route)));
    if (!file.startsWith(root+path.sep) || !fs.statSync(file).isFile()) {res.writeHead(404);return res.end('Not found');}
    res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','Cache-Control':'no-cache','X-Content-Type-Options':'nosniff'});
    if(req.method==='HEAD')return res.end();
    fs.createReadStream(file).pipe(res);
  } catch {res.writeHead(404);res.end('Not found');}
}).listen(8787,'127.0.0.1',()=>console.log('Lecture server: http://127.0.0.1:8787'));

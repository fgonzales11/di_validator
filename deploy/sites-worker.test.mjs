import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// Import the deployable ES module without depending on the parent project's package type.
const source = await readFile(new URL('./sites-worker.js', import.meta.url), 'utf8');
const {default: worker} = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const originalFetch = globalThis.fetch;
let captured;
globalThis.fetch = async (url, options) => {
  captured = {url,options};
  return new Response('{"datasets":1}', {headers:{'content-type':'application/json','set-cookie':'private=upstream'}});
};
try {
  const env = {DI_BACKEND_URL:'https://backend.example',DI_GATEWAY_TOKEN:'test-secret',DI_API_KEY:'test-api-key',
    ASSETS:{fetch: async req => new Response(new URL(req.url).pathname)}};
  const response = await worker.fetch(new Request('https://site.example/api/v1/overview?limit=1',
    {headers:{cookie:'site-session=private',authorization:'Bearer browser','x-di-gateway-token':'forged',
      'x-api-key':'forged','oai-sites-authorization':'private-site-token','x-serverless-authorization':'forged'}}),env);
  assert.equal(captured.url.href,'https://backend.example/api/v1/overview?limit=1');
  assert.equal(captured.options.headers.get('x-di-gateway-token'),'test-secret');
  assert.equal(captured.options.headers.get('x-api-key'),'test-api-key');
  assert.equal(captured.options.headers.has('oai-sites-authorization'),false);
  assert.equal(captured.options.headers.has('x-serverless-authorization'),false);
  assert.equal(captured.options.headers.has('cookie'),false);
  assert.equal(captured.options.headers.has('authorization'),false);
  assert.equal(response.headers.has('set-cookie'),false);
  assert.equal(response.headers.get('cache-control'),'no-store');
  assert.equal(response.headers.get('cross-origin-opener-policy'),'same-origin');
  assert.equal(response.headers.get('cross-origin-embedder-policy'),'require-corp');
  assert.equal(await response.text(),'{"datasets":1}');
  assert.equal((await worker.fetch(new Request('https://site.example/api/v1/datasets'),{})).status,503);
  assert.equal(await (await worker.fetch(new Request('https://site.example/assets/chart.js'),env)).text(),'/assets/chart.js');
  env.ASSETS.fetch = async req => new URL(req.url).pathname==='/index.html'
    ? new Response(null,{status:301,headers:{location:'/'}})
    : new Response('index', {status:new URL(req.url).pathname==='/'?200:404});
  assert.equal((await worker.fetch(new Request('https://site.example/events',{headers:{accept:'text/html'}}),env)).status,200);
  globalThis.fetch = async () => {throw new Error('Transport failure')};
  assert.equal((await worker.fetch(new Request('https://site.example/api/v1/health'),env)).status,502);
  console.log('Sites gateway authentication, streaming response headers, routing and failures verified.');
} finally {
  globalThis.fetch=originalFetch;
}

import {mkdir,copyFile,cp,writeFile} from 'node:fs/promises';
await mkdir('dist/server', {recursive: true});
await mkdir('dist/.openai', {recursive: true});
await cp('frontend/dist', 'dist/client', {recursive: true});
await copyFile('.openai/hosting.json', 'dist/.openai/hosting.json');
await copyFile('deploy/sites-worker.js', 'dist/server/index.js');
await writeFile('dist/server/wrangler.json', JSON.stringify({
  name: 'di-validator', main: 'index.js', compatibility_date: '2026-09-20',
  compatibility_flags: ['nodejs_compat'],
  assets: {directory: '../client', binding: 'ASSETS', run_worker_first: true},
}));

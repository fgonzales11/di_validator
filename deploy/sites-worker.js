// Runs on Sites, behind the site's access controls. Never send the secret to JS clients.
export default {
  async fetch(request, env) {
    const response = await serve(request, env);
    const headers = new Headers(response.headers);
    // JupyterLite uses shared memory to expose notebook files to its Python kernel.
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Resource-Policy', 'same-origin');
    return new Response(response.body, {status: response.status, statusText: response.statusText, headers});
  }
};

async function serve(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/notebooks/') ||
        ['/docs', '/openapi.json', '/redoc'].includes(url.pathname)) {
      if (!env.DI_BACKEND_URL || !env.DI_GATEWAY_TOKEN || !env.DI_API_KEY) {
        return Response.json({detail: 'Backend connection is not configured.'}, {status: 503});
      }
      const target = new URL(env.DI_BACKEND_URL);
      target.pathname = url.pathname;
      target.search = url.search;
      const headers = new Headers(request.headers);
      headers.delete('cookie');
      headers.delete('authorization');
      headers.delete('oai-sites-authorization');
      headers.delete('x-serverless-authorization');
      headers.delete('host');
      headers.set('x-di-gateway-token', env.DI_GATEWAY_TOKEN);
      headers.set('x-api-key', env.DI_API_KEY);
      headers.set('x-forwarded-host', url.host);
      headers.set('x-forwarded-proto', 'https');
      try {
        const upstream = await fetch(target, {
          method: request.method, headers, redirect: 'manual',
          body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
        });
        const responseHeaders = new Headers(upstream.headers);
        responseHeaders.delete('set-cookie');
        responseHeaders.delete('x-di-gateway-token');
        if (url.pathname.startsWith('/api/')) responseHeaders.set('cache-control', 'no-store');
        const redirect = responseHeaders.get('location');
        if (redirect?.startsWith(env.DI_BACKEND_URL)) {
          responseHeaders.set('location', redirect.replace(env.DI_BACKEND_URL, url.origin));
        }
        return new Response(upstream.body, {status: upstream.status, headers: responseHeaders});
      } catch {
        return Response.json({detail: 'The analysis service is temporarily unavailable.'}, {status: 502});
      }
    }
    const asset = await env.ASSETS.fetch(request);
    if (asset.status !== 404 || !request.headers.get('accept')?.includes('text/html')) return asset;
    // The asset service canonicalizes /index.html to / with a redirect.
    // Fetch / directly so the browser retains its requested application route.
    return env.ASSETS.fetch(new Request(new URL('/', url), request));
}

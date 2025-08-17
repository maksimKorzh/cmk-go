self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('my-app-cache').then((cache) => {
      return cache.addAll([
        '/cmkgo/',
        '/cmkgo/manifest.json',
        '/cmkgo/index.html',
        '/cmkgo/tfjs/tf.js',
        '/cmkgo/model/model.json',
        '/cmkgo/model/group1-shard1of4.bin',
        '/cmkgo/model/group1-shard2of4.bin',
        '/cmkgo/model/group1-shard3of4.bin',
        '/cmkgo/model/group1-shard4of4.bin',
        '/cmkgo/assets/icon-192x192.png',
        '/cmkgo/assets/icon-512x512.png'
      ]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});

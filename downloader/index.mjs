import fetch from 'node-fetch';
import { writeFile, open } from 'fs/promises';
import pMap from 'p-map';
import ProgressBar from 'progress';
import { createHash } from 'crypto';
import { promisify } from 'util';
import {
  brotliDecompress as brotliDecompressWithCb,
  gunzip as gunzipWithCb
} from 'zlib';
import AbortController from 'abort-controller';
import csvStringify from 'csv-stringify/lib/sync.js';
import { createReadStream } from 'fs';
import csvParse from 'csv-parse';

process.env['NODE_TLS_REJECT_UNAUTHORIZED'] = '0';

const brotliDecompress = promisify(brotliDecompressWithCb);
const gunzip = promisify(gunzipWithCb);

let urls = new Set();

for await (let { url } of createReadStream('wasms.csv', 'utf-8').pipe(
  csvParse({ columns: true })
)) {
  urls.add(url);
}

let resultsFile = await open('results.csv', 'a+');

let existingResultsText = await resultsFile.readFile('utf-8');

if (!existingResultsText) {
  await resultsFile.write(`url,raw_size,size,filename\n`);
} else {
  for await (let { url } of csvParse(existingResultsText, { columns: true })) {
    urls.delete(url);
  }
}

let progress = new ProgressBar(':bar :current/:total :etas left', {
  total: urls.size
});

/** @param {Buffer} data */
function isWasm(data) {
  return data.subarray(0, 4).toString('binary') === '\0asm';
}

try {
  await pMap(
    urls,
    async url => {
      try {
        let controller = new AbortController();
        setTimeout(() => {
          controller.abort();
        }, 120000);
        let { signal } = controller;

        let res = await fetch(url, {
          signal,
          headers: {
            'Accept-Encoding': 'gzip, br',
            'User-Agent':
              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36'
          },
          compress: false
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status} ${res.statusText}`);
        }
        let data = Buffer.from(await res.arrayBuffer());
        let rawDataSize = data.length;

        let looksLikeWasm = isWasm(data);

        // In some cases Content-Encoding seems not specified correctly.
        // Just try known compression algorithms and see if they work.
        if (!looksLikeWasm) {
          let couldDecompress = false;
          for (let algo of [gunzip, brotliDecompress]) {
            try {
              data = await algo(data);
              couldDecompress = true;
              break;
            } catch {}
          }
          if (!couldDecompress || !isWasm(data)) {
            throw new Error('Not WebAssembly');
          }
        }

        let hash = createHash('sha256');
        hash.update(data);
        let filename = `${hash.digest('hex')}.wasm`;

        await resultsFile.write(csvStringify([[url, rawDataSize, data.length, filename]]));

        try {
          await writeFile(filename, data, { flag: 'wx' });
        } catch (e) {
          if (e.code !== 'EEXIST') throw e;
        }
      } catch (e) {
        e.message += ` in ${url}`;
        throw e;
      } finally {
        progress.tick();
      }
    },
    { concurrency: 10, stopOnError: false }
  );
} finally {
  await resultsFile.close();
}

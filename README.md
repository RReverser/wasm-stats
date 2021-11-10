# wasm-stats

A  tool used for WebAssembly analysis in Web Almanac.

It takes a directory with WebAssembly files as its only CLI argument:

```bash
$ cargo run --release -- wasms-dir
```

There, it walks over all the `.wasm` files (`wasms-dir/*.wasm`) and collects all the revelant stats - instruction kinds, section sizes, numbers of imports/exports, etc.

The results are stored in a newline-delimited JSON file `stats.json` in the same provided directory (`wasms-dir/stats.json`) that can be later imported in a database like BigQuery for further analysis.

If execution was stopped midway for any reason, the next rerun of the tool will automatically skip any files that have been already found in `stats.json` and resume analysis.

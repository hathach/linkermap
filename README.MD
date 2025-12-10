# linkermap

Analyze GNU/Clang/IAR ld’s linker map, based on [visual version](https://github.com/PromyLOPh/linkermapviz) .
Works best with `-ffunction-sections -fdata-sections` and statically linked code (e.g. for bare-metal embedded systems).
Note: `-flto` merges code across .o files and make it difficult to analyze.

## Install

```bash
$ pip install .
$ linkermap --version
0.5.0
```

## Usages

```bash
$ linkermap --help (or python linker-map.py --help)
usage: linkermap [-h] [-v] [-s EXTRA_SECTIONS] [-j] [-f FILTERS] [-m] [-S SORT] [-q] [-d OUT_DIR] [-V] map_file

Analyze GNU/Clang/IAR ld linker map.

positional arguments:
  map_file              Path to the linker map file to analyze.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print symbols within file.
  -s EXTRA_SECTIONS, --section EXTRA_SECTIONS
                        Additional section name to include; repeat for multiple sections.
  -j, --json            Write JSON summary next to the map file.
  -f FILTERS, --filter FILTERS
                        Only include object files whose path contains this substring (can be repeated).
  -m, --markdown        Write Markdown table next to the map file.
  -S SORT, --sort SORT  Sorting: size-, size+, name-, name+ (default size-). Applies to stdout and markdown.
  -q, --quiet           Suppress standard summary output.
  -d OUT_DIR, --dir OUT_DIR
                        Directory to write JSON/Markdown outputs (default: alongside map file).
  -V, --version         show program's version number and exit
```

### non-verbose

```bash
$ linkermap --sort=size- cdc_msc.gnu.map
| File                      |    .text |  .rodata |    .data |     .bss |     size |      % |
| :------------------------ | -------: | -------: | -------: | -------: | -------: | -----: |
| msc_disk.o                |      300 |       13 |     8192 |        1 |     8506 |  34.3% |
| usbd.o                    |     2718 |       48 |       24 |      245 |     3035 |  12.2% |
| dcd_synopsys.o            |     2578 |        0 |        0 |      143 |     2721 |  11.0% |
| ...                       |      ... |      ... |      ... |      ... |      ... |    ... |
| TOTAL                     |    14774 |      257 |     8253 |     1502 |    24786 | 100.0% |

```

### verbose

```
$ linkermap -sort=size- -v cdc_msc.gnu.map
Each file is printed as its own table (file row + symbols), with the file row showing size and %. Symbol rows omit % and per-file totals replace the old SUM row.
```

### Markdown output

```bash
$ linkermap --sort=size- -m cdc_msc.elf.map

# Linker Map Summary
| File                      |    .text |  .rodata |    .data |     .bss |     size |      % |
| :------------------------ | -------: | -------: | -------: | -------: | -------: | -----: |
| msc_disk.o                |      300 |       13 |     8192 |        1 |     8506 |  34.3% |
| usbd.o                    |     2718 |       48 |       24 |      245 |     3035 |  12.2% |
| ...                       |      ... |      ... |      ... |      ... |      ... |    ... |
| TOTAL                     |    14774 |      257 |     8253 |     1502 |    24786 | 100.0% |
```

**extra sections & JSON/Markdown output**

Enable JSON/Markdown outputs with flags; files are written beside the map using the map’s basename:

```bash
$ linkermap -s .fastcode -s .itcm -j -m build/firmware.map
```

Tables now include `size` and `%` columns; the final row is `TOTAL` (no `%` in symbol rows). Sections are discovered from the map and sorted (now reverse alphabetical). JSON contains `files` (with `size`, `percent`, `sections`), and a top-level `TOTAL`; verbose output produces one table per file with its symbols underneath. Use `-f/--filter` (repeatable) to include only object files whose path contains given substrings.

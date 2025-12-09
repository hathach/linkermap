__all__ = ["main", "analyze_map", "write_json", "write_markdown", "version_str"]
# vim: set fileencoding=utf8 :

import argparse
import json
import sys, re, os
from itertools import groupby

# Avoid deprecated pkg_resources; prefer stdlib importlib.metadata.
try:  # Python >=3.8
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore

try:
    version_str = importlib_metadata.version("linkermap")
except importlib_metadata.PackageNotFoundError:
    version_str = "0.0.0"


class Objectfile:
    def __init__ (self, section, offset, size, comment):
        self.section = section.strip ()
        self.offset = offset
        self.size = size
        self.path = (None, None)
        self.basepath = None
        if comment:
            self.path = re.match(r'^(.+?)(?:\(([^)]+)\))?$', comment).groups()
            self.basepath = os.path.basename(self.path[0])
            if self.basepath.endswith(":"):
                self.basepath = self.basepath[:-1]
            # Normalize compiler-generated suffixes like .c.o / .c.obj back to .c
            if self.basepath.endswith('.c.o'):
                self.basepath = self.basepath[:-2]  # drop trailing .o
            elif self.basepath.endswith('.c.obj'):
                self.basepath = self.basepath[:-4]  # drop trailing .obj
        self.children = []

    def __repr__ (self):
        return '<Objectfile {} {:x} {:x} {} {}>'.format (self.section, self.offset, self.size, self.path, repr (self.children))


def parse_clang_map(lines):
    sections = []
    current_section = None
    last_object = None

    for line in lines:
        if not line.strip():
            continue
        if line.lstrip().startswith('VMA'):
            continue

        tokens = line.split()
        if len(tokens) < 4:
            continue

        try:
            vma = int(tokens[0], 16)
            size = int(tokens[2], 16)
        except ValueError:
            continue

        out_field = tokens[4] if len(tokens) >= 5 else ''

        # New section line
        if out_field.startswith('.') and out_field != '.':
            current_section = out_field
            sec_obj = Objectfile(current_section, vma, size, comment=None)
            sections.append(sec_obj)
            last_object = None
            continue

        if current_section is None:
            continue

        # Object contribution line
        if len(tokens) >= 5 and ':' in out_field:
            comment = ' '.join(tokens[4:])
            last_object = Objectfile(current_section, vma, size, comment=comment)
            sections[-1].children.append(last_object)
            continue

        # Symbol within last object
        if last_object is not None and len(tokens) >= 5:
            symbol_name = ' '.join(tokens[4:])
            try:
                sym_offset = int(tokens[0], 16)
            except ValueError:
                sym_offset = last_object.offset
            last_object.children.append((sym_offset, symbol_name))

    return sections


def parse_gnu_map(lines):
    sections = []

    try:
        mem_idx = next(i for i, l in enumerate(lines) if l.strip() == 'Memory Configuration')
    except StopIteration:
        return None

    content = ''.join(lines[mem_idx + 1:])

    sectionre = re.compile('(?P<section>.+?|.{14,}\n)[ ]+0x(?P<offset>[0-9a-f]+)[ ]+0x(?P<size>[0-9a-f]+)(?:[ ]+(?P<comment>.+))?\n+', re.I)
    subsectionre = re.compile('[ ]{16}0x(?P<offset>[0-9a-f]+)[ ]+(?P<function>.+)\n+', re.I)
    s = content
    pos = 0
    while True:
        m = sectionre.match(s, pos)
        if not m:
            try:
                nextpos = s.index('\n', pos) + 1
                pos = nextpos
                continue
            except ValueError:
                break
        pos = m.end()
        section = m.group('section')
        v = m.group('offset')
        offset = int(v, 16) if v is not None else None
        v = m.group('size')
        size = int(v, 16) if v is not None else None
        comment = m.group('comment')
        if section != '*default*' and size > 0:
            of = Objectfile(section, offset, size, comment)
            if section.startswith(' '):
                sections[-1].children.append(of)
                while True:
                    m = subsectionre.match(s, pos)
                    if not m:
                        break
                    pos = m.end()
                    offset, function = m.groups()
                    offset = int(offset, 16)
                    if sections and sections[-1].children:
                        sections[-1].children[-1].children.append((offset, function))
            else:
                sections.append(of)
    return sections


def parse_iar_map(lines):
    sections = {}
    offsets = {'.text': 0, '.rodata': 0, '.data': 0}
    try:
        start_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('*** MODULE SUMMARY'))
    except StopIteration:
        return None

    path_line_prefix = ''
    entry_re_cols = re.compile(r'^\s+([^\s]+)\s+([0-9\'\"]+)(?:\s+([0-9\'\"]+))?(?:\s+([0-9\'\"]+))?\s*$')

    def section_obj(name):
        if name not in sections:
            sections[name] = Objectfile(name, 0, 0, comment=None)
        return sections[name]

    for line in lines[start_idx + 1:]:
        if not line.strip():
            continue
        if line.startswith('***'):
            continue
        if line.startswith('----'):
            continue
        if line.lstrip().startswith('Total:') or line.lstrip().startswith('Gaps') or line.lstrip().startswith('Linker created'):
            continue
        if line.startswith('    -------------------------------------------------'):
            continue
        if not line.startswith('    '):
            if ': [' in line:
                path_line_prefix = line.split(':', 1)[0].strip()
            continue

        m = entry_re_cols.match(line)
        if not m:
            continue
        obj_name, ro_code, ro_data, rw_data = m.groups()

        def to_int(val):
            if not val:
                return 0
            return int(val.replace("'", '').replace('"', ''))

        ro_code_i = to_int(ro_code)
        ro_data_i = to_int(ro_data)
        rw_data_i = to_int(rw_data)

        full_path = os.path.join(path_line_prefix, obj_name) if path_line_prefix else obj_name

        if ro_code_i:
            sec = section_obj('.text')
            obj = Objectfile('.text', offsets['.text'], ro_code_i, full_path)
            sec.size += ro_code_i
            sec.children.append(obj)
            offsets['.text'] += ro_code_i
        if ro_data_i:
            sec = section_obj('.rodata')
            obj = Objectfile('.rodata', offsets['.rodata'], ro_data_i, full_path)
            sec.size += ro_data_i
            sec.children.append(obj)
            offsets['.rodata'] += ro_data_i
        if rw_data_i:
            sec = section_obj('.data')
            obj = Objectfile('.data', offsets['.data'], rw_data_i, full_path)
            sec.size += rw_data_i
            sec.children.append(obj)
            offsets['.data'] += rw_data_i

    return list(sections.values())

def parseSections (fd):
    """
    Parse GNU ld, clang/LLVM, or IAR map files.
    """

    first_line = fd.readline()
    rest = fd.readlines()
    lines = [first_line, *rest]

    if 'VMA' in first_line and 'LMA' in first_line and 'Size' in first_line:
        return parse_clang_map(lines)

    if any('IAR ELF Linker' in l for l in lines[:10]):
        return parse_iar_map(lines)

    return parse_gnu_map(lines)



def print_file(verbose, symlist, format_row, sep_line, tail=True, emit=print, first_sep=True, percent_lookup=None):
    for idx, sym in enumerate(symlist):
        name = sym[0]
        if idx != 0:
            name += '  '
        values = list(sym[1].values())
        size_val = sum(values)
        percent_val = percent_lookup(sym[0], size_val, idx == 0) if percent_lookup else ''
        emit(format_row(values, size_val, percent_val, name))
        if verbose and idx == 0 and first_sep:
            emit(sep_line)

    if verbose and tail:
        emit(sep_line)


def _parse_sort_opt(value):
    """Return (field, reverse) tuple from sort option string.

    Accepts size+, size-, name+, name-. Defaults to name+.
    """
    if value is None:
        value = 'name+'
    if isinstance(value, tuple) and len(value) == 2:
        field, reverse = value
        return (field, reverse)
    val = str(value).lower()
    mapping = {
        'size-': ('size', True),
        'size+': ('size', False),
        'size': ('size', False),
        'name-': ('name', True),
        'name+': ('name', False),
        'name': ('name', False),
    }
    if val not in mapping:
        val = 'name+'
    return mapping[val]


def _render_table(json_data, verbose, sort_opt, emit_line):
    files = json_data["files"]

    # Build section list on the fly, sorted alphabetically.
    section_set = set()
    for f in files:
        section_set.update(f["sections"].keys())
    section_list = sorted(section_set, reverse=True)

    # Dynamic widths based on longest names.
    name_candidates = [f["file"] for f in files]
    if verbose:
        for f in files:
            for sec_val in f["sections"].values():
                if isinstance(sec_val, dict):
                    name_candidates.extend(sec_val.keys())
    name_width = max(len(n) for n in name_candidates + ['TOTAL', 'File'])

    col_width = max([8, len("size"), *(len(sec) for sec in section_list)])
    col_fmt = '{:' + f'>{col_width}' + '}'
    file_fmt = '{:' + f'<{name_width}' + '}'
    percent_width = max(len('%'), len('100.0%'))
    percent_fmt = '{:' + f'>{percent_width}' + '}'

    def format_percent(val):
        if val == '' or val is None:
            return ''
        return f"{val:.1f}%"

    def format_row(section_vals, size_val, percent_val, fname):
        cols = [file_fmt.format(fname)]
        cols.extend(col_fmt.format(v) for v in section_vals)
        cols.append(col_fmt.format(size_val))
        cols.append(percent_fmt.format(format_percent(percent_val) if isinstance(percent_val, (int, float)) else percent_val))
        return '| ' + ' | '.join(cols) + ' |'

    def _align_right(width):
        return '-' * max(3, width - 1) + ':'  # Markdown right align

    def _align_left(width):
        return ':' + '-' * max(2, width - 1)  # Markdown left align

    sep_line = '| ' + ' | '.join([_align_left(name_width)] + [_align_right(col_width) for _ in section_list] + [_align_right(col_width)] + [_align_right(percent_width)]) + ' |'

    sort_field, reverse = _parse_sort_opt(sort_opt)
    file_key = (lambda x: x['size']) if sort_field == "size" else (lambda x: x['file'].lower())

    sum_all = dict.fromkeys(section_list, 0)
    sum_all['total'] = 0

    files_sorted = sorted(files, key=file_key, reverse=reverse)

    if not verbose:
        header = format_row(section_list, 'size', '%', 'File')
        emit_line(header)
        emit_line(sep_line)

    for idx, f in enumerate(files_sorted):
        fname = f["file"]
        finfo = {fname: dict.fromkeys(section_list, 0)}
        for s in section_list:
            if s in f["sections"]:
                sec_val = f["sections"][s]
                if verbose and isinstance(sec_val, dict):
                    for sym, size in sec_val.items():
                        finfo[sym] = dict.fromkeys(section_list, 0)
                        finfo[sym][s] = size
                    size = sum(sec_val.values())
                else:
                    size = sec_val
                finfo[fname][s] = size
                sum_all[s] += size
                sum_all['total'] += size

        sym_key = (lambda x: sum(x[1].values())) if sort_field == "size" else (lambda x: x[0].lower())
        items = list(finfo.items())
        if verbose:
            file_entry = next((i for i in items if i[0] == fname), None)
            symbol_entries = [i for i in items if i[0] != fname]
            symbol_entries_sorted = sorted(symbol_entries, key=sym_key, reverse=reverse)
            symlist_sorted = [file_entry, *symbol_entries_sorted] if file_entry else symbol_entries_sorted
        else:
            symlist_sorted = sorted(items, key=sym_key, reverse=reverse)

        percent_lookup = lambda n, size_val, is_file: f.get("percent", 0) if is_file else ''

        if verbose:
            # Per-file table with header and separator, then file row followed by symbols.
            header = format_row(section_list, 'size', '%', 'File')
            emit_line(header)
            emit_line(sep_line)
            print_file(True, symlist_sorted, format_row, sep_line, tail=False, emit=emit_line, first_sep=False, percent_lookup=percent_lookup)
            emit_line("")  # blank line between tables
        else:
            is_last_file = idx == len(files_sorted) - 1
            print_file(False, symlist_sorted, format_row, sep_line, tail=not is_last_file, emit=emit_line, percent_lookup=percent_lookup)

    if not verbose:
        # Sum row only for non-verbose table.
        section_sums = [sum_all[sec] for sec in section_list]
        emit_line(format_row(section_sums, sum_all['total'], 100.0, 'TOTAL'))


def print_summary(json_data, verbose, sort_opt="size-"):
    _render_table(json_data, verbose, sort_opt, print)

def build_parser():
    parser = argparse.ArgumentParser(description='Analyze GNU/Clang/IAR ld linker map.')
    parser.add_argument('map_file', help='Path to the linker map file to analyze.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print symbols within file.')
    parser.add_argument(
        '-s', '--section',
        action='append', dest='extra_sections', default=[],
        help='Additional section name to include; repeat for multiple sections.'
    )
    parser.add_argument(
        '-j', '--json',
        dest='json_out',
        action='store_true',
        help='Write JSON summary next to the map file.'
    )
    parser.add_argument(
        '-f', '--filter',
        dest='filters',
        action='append',
        default=[],
        help='Only include object files whose path contains this substring (can be repeated).'
    )
    parser.add_argument(
        '-m', '--markdown',
        dest='markdown_out',
        action='store_true',
        help='Write Markdown table next to the map file.'
    )
    parser.add_argument(
        '-S', '--sort',
        dest='sort',
        default='size-',
        help="Sorting: size-, size+, name-, name+ (default size-). Applies to stdout and markdown."
    )
    parser.add_argument(
        '-q', '--quiet',
        dest='quiet',
        action='store_true',
        help='Suppress standard summary output.'
    )
    parser.add_argument(
        '-d', '--dir',
        dest='out_dir',
        help='Directory to write JSON/Markdown outputs (default: alongside map file).'
    )
    parser.add_argument('-V', '--version', action='version', version=version_str)
    return parser



def analyze_map(map_file, verbose=False, filters=None, extra_sections=None):
    filters = filters or []
    extra_sections = extra_sections or []

    fd = open(map_file, encoding='utf-8')
    all_sections = parseSections(fd)
    if all_sections is None:
        raise RuntimeError('start of memory config not found, did you invoke the compiler/linker with LANG=C?')

    base_sections = ['.text', '.rodata', '.data', '.bss', *extra_sections]
    wanted_sections = [sec for sec in all_sections if sec.section in base_sections]

    files_out = []
    for s in wanted_sections:
        objects = s.children
        # Apply path filters if provided.
        if filters:
            objects = [o for o in objects if o.path[0] and any(filt in o.path[0] for filt in filters)]
        for k, g in groupby(sorted(objects, key=lambda x: x.basepath), lambda x: x.basepath):
            group_list = list(g)
            if not group_list:
                continue
            entry = next((f for f in files_out if f["file"] == k), None)
            if not entry:
                entry = {"file": k, "path": group_list[0].path[0], "sections": {}, "size": 0}
                files_out.append(entry)
            entry.setdefault("sections", {}).setdefault(s.section, {})
            for symbol in sorted(group_list, reverse=True, key=lambda x: x.size):
                entry["sections"][s.section][symbol.children[0][1] if symbol.children else symbol.section] = symbol.size
                entry["size"] += symbol.size

    if not verbose:
        # collapse section dictionaries to totals
        for f in files_out:
            collapsed = {sec: sum(f["sections"][sec].values()) for sec in f["sections"]}
            f["sections"] = collapsed

    total_size = sum(f["size"] for f in files_out)
    for f in files_out:
        f["percent"] = round((f["size"] / total_size) * 100, 2) if total_size else 0.0

    json_data = {
        "files": files_out,
        "TOTAL": total_size,
    }

    return json_data

def write_json(json_data, path):
    with open(path, "w", encoding="utf-8") as outf:
        json.dump(json_data, outf, indent=2)


def write_markdown(json_data, path, verbose=False, sort_opt="size-", title="Linker Map Summary"):
    md_lines = [f"# {title}", ""]

    lines: list[str] = []
    _render_table(json_data, verbose, sort_opt, lines.append)

    if lines:
        md_lines.extend(lines)
    else:
        md_lines.append("(no matching object files)")
    md_lines.append("")

    # Build mapfiles list after the summary table inside a collapsible block
    if "mapfiles" in json_data and json_data["mapfiles"]:
        mapfiles_lines = [
            "<details>",
            "<summary>Map Files</summary>",
            "",
        ]
        for mf in json_data["mapfiles"]:
            mapfiles_lines.append(f"- {mf}")
        mapfiles_lines.append("")
        mapfiles_lines.append("</details>")
        mapfiles_lines.append("")
        if md_lines and md_lines[-1] != "":
            md_lines.append("")
        md_lines.extend(mapfiles_lines)

    with open(path, "w", encoding="utf-8") as mdfile:
        mdfile.write("\n".join(md_lines))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    map_file = args.map_file
    sort_opt = args.sort
    verbose = args.verbose
    filters = args.filters or []
    extra_sections = args.extra_sections or []
    want_json = args.json_out
    want_markdown = args.markdown_out
    out_dir = args.out_dir
    quiet = args.quiet

    base_name = os.path.basename(map_file)
    target_dir = out_dir if out_dir else os.path.dirname(map_file)
    if not target_dir:
        target_dir = "."
    json_fname = os.path.join(target_dir, base_name + ".json")
    markdown_fname = os.path.join(target_dir, base_name + ".md")

    json_data = analyze_map(map_file, verbose, filters, extra_sections)

    if not quiet:
        print_summary(json_data, verbose, sort_opt)

    if want_json:
        write_json(json_data, json_fname)
        if not quiet:
            print(f"JSON summary written to {json_fname}")

    if want_markdown:
        write_markdown(json_data, markdown_fname, verbose, sort_opt)
        if not quiet:
            print(f"Markdown summary written to {markdown_fname}")


if __name__ == '__main__':
    main()

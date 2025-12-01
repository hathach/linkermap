# vim: set fileencoding=utf8 :

import argparse
import json
import sys, re, os
from itertools import chain, groupby
import pandas as pd

# Avoid deprecated pkg_resources; prefer stdlib importlib.metadata.
try:  # Python >=3.8
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python <3.8 fallback
    import importlib_metadata  # type: ignore

try:
    version_str = importlib_metadata.version("linkermap")
except importlib_metadata.PackageNotFoundError:
    version_str = "0.0.0"

sfmt = '{:>8}'


class Objectfile:
    def __init__ (self, section, offset, size, comment):
        self.section = section.strip ()
        self.offset = offset
        self.size = size
        self.path = (None, None)
        self.basepath = None
        if comment:
            self.path = re.match (r'^(.+?)(?:\(([^\)]+)\))?$', comment).groups ()
            self.basepath = os.path.basename (self.path[0])
        self.children = []

    def __repr__ (self):
        return '<Objectfile {} {:x} {:x} {} {}>'.format (self.section, self.offset, self.size, self.path, repr (self.children))


def parseSections (fd):
    """
    Quick&Dirty parsing for GNU ld’s linker map output, needs LANG=C, because
    some messages are localized.
    """

    sections = []

    # skip until memory map is found
    found = False
    while True:
        l = fd.readline()
        if not l:
            break
        if l.strip () == 'Memory Configuration':
            found = True
            break
    if not found:
        return None

    # long section names result in a linebreak afterwards
    sectionre = re.compile ('(?P<section>.+?|.{14,}\n)[ ]+0x(?P<offset>[0-9a-f]+)[ ]+0x(?P<size>[0-9a-f]+)(?:[ ]+(?P<comment>.+))?\n+', re.I)
    subsectionre = re.compile ('[ ]{16}0x(?P<offset>[0-9a-f]+)[ ]+(?P<function>.+)\n+', re.I)
    s = fd.read ()
    pos = 0
    while True:
        m = sectionre.match (s, pos)
        if not m:
            # skip that line
            try:
                nextpos = s.index ('\n', pos)+1
                pos = nextpos
                continue
            except ValueError:
                break
        pos = m.end ()
        section = m.group ('section')
        v = m.group ('offset')
        offset = int (v, 16) if v is not None else None
        v = m.group ('size')
        size = int (v, 16) if v is not None else None
        comment = m.group ('comment')
        if section != '*default*' and size > 0:
            of = Objectfile (section, offset, size, comment)
            if section.startswith (' '):
                sections[-1].children.append (of)
                while True:
                    m = subsectionre.match (s, pos)
                    if not m:
                        break
                    pos = m.end ()
                    offset, function = m.groups ()
                    offset = int (offset, 16)
                    if sections and sections[-1].children:
                        sections[-1].children[-1].children.append ((offset, function))
            else:
                sections.append (of)

    return sections


def print_file(verbose, header, symlist, ffmt):
    for sym in symlist:
        n = sym[0]
        if symlist.index(sym) != 0:
            n += ' ' * 2
        print(ffmt.format(n) + ''.join(map(sfmt.format, sym[1].values())) + sfmt.format(sum(sym[1].values())))
        if verbose and symlist.index(sym) == 0:
            spaces = ffmt.format('-'*len(n))
            print(spaces + '-' * (len(header) - len(spaces)))

    if verbose:
        print('-' * len(header))


def print_summary(verbose, section_list, symbol_table):
    # Dynamic right-aligned width based on longest file/symbol name.
    name_candidates = list(symbol_table.keys())
    if verbose:
        for f in symbol_table.values():
            for sec, syms in f.items():
                if sec in {'total', 'path'}:
                    continue
                name_candidates.extend(syms.keys())
    name_width = max(len(n) for n in name_candidates + ['SUM', 'File'])
    ffmt = '{:' + f'>{name_width}' + '} |'

    header = ffmt.format('File') + ''.join(map(sfmt.format, section_list)) + sfmt.format('Total')
    print(header)
    print('-'*len(header))

    sum_all = dict.fromkeys(section_list, 0)
    sum_all['total'] = 0

    for file in sorted(symbol_table, key=lambda x: symbol_table[x]['total'], reverse=True):
        # Print overall of a file object
        finfo = {file: dict.fromkeys(section_list, 0)}
        for s in section_list:
            if s in symbol_table[file]:
                if verbose:
                    for sym in symbol_table[file][s]:
                        finfo[sym] = dict.fromkeys(section_list, 0)
                        finfo[sym][s] = symbol_table[file][s][sym]

                size = sum(symbol_table[file][s].values())
                finfo[file][s] = size

                sum_all[s] += size
                sum_all['total'] += size
        print_file(verbose, header, sorted(finfo.items(), key=lambda x: sum(x[1].values()), reverse=True), ffmt)

    # Sum
    print(ffmt.format('SUM') + ''.join(map(sfmt.format, sum_all.values())))


def build_parser():
    parser = argparse.ArgumentParser(description='Analyze GNU ld linker map.')
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
        '-q', '--quiet',
        dest='quiet',
        action='store_true',
        help='Suppress standard summary output.'
    )
    parser.add_argument('-V', '--version', action='version', version=version_str)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    map_file = args.map_file
    verbose = args.verbose
    filters = args.filters or []
    extra_sections = args.extra_sections or []
    want_json = args.json_out
    want_markdown = args.markdown_out
    json_fname = map_file + ".json"
    markdown_fname = map_file + ".md"
    quiet = args.quiet

    fd = open(map_file)
    sections = parseSections (fd)
    if sections is None:
        print ('start of memory config not found, did you invoke the compiler/linker with LANG=C?')
        return

    sectionWhitelist = {'.text', '.data', '.bss', '.rodata', *extra_sections}
    whitelistedSections = list (filter (lambda x: x.section in sectionWhitelist, sections))
    #allObjects = list (chain (*map (lambda x: x.children, whitelistedSections)))
    #allFiles = list (set (map (lambda x: x.basepath, allObjects)))

    symbol_table = {}

    for s in whitelistedSections:
        objects = s.children
        # Apply path filters if provided.
        if filters:
            objects = [o for o in objects if o.path[0] and any(filt in o.path[0] for filt in filters)]
        for k, g in groupby(sorted(objects, key=lambda x: x.basepath), lambda x: x.basepath):
            group_list = list(g)
            if not group_list:
                continue
            symbol_table.setdefault(k, {})
            symbol_table[k].setdefault('total', 0)
            symbol_table[k].setdefault(s.section, {})
            # record source path once
            if 'path' not in symbol_table[k]:
                symbol_table[k]['path'] = group_list[0].path[0]
            for symbol in sorted(group_list, reverse=True, key=lambda x: x.size):
                symbol_table[k][s.section][symbol.children[0][1] if symbol.children else symbol.section] = symbol.size
                symbol_table[k]['total'] += symbol.size

    # Persist JSON-ready structure (also used for printing).
    json_sections = list(map(lambda x: x.section, whitelistedSections))
    json_data = {
        "sections": json_sections,
        "verbose": verbose,
        "files": []
    }
    for fname, fdata in symbol_table.items():
        if verbose:
            section_entries = {s: fdata[s] for s in json_sections if s in fdata}
        else:
            section_entries = {s: sum(fdata[s].values()) for s in json_sections if s in fdata}
        json_data["files"].append({
            "file": fname,
            "total": fdata.get("total", 0),
            "sections": section_entries,
            "path": symbol_table[fname].get("path")
        })

    if not quiet:
        print_summary(verbose, json_sections, symbol_table)

    if want_json:
        with open(json_fname, 'w', encoding='utf-8') as outf:
            json.dump(json_data, outf, indent=2)
        if not quiet:
            print(f'JSON summary written to {json_fname}')

    if want_markdown:
        rows = []
        if verbose:
            md_lines = ["# Linker Map Summary", "", f"Sections included: {', '.join(json_sections)}", ""]
            # build nested tables: one per file
            files_sorted = sorted(json_data["files"], key=lambda f: f["total"], reverse=True)
            for f in files_sorted:
                rows = []
                for section_name, symbols in f["sections"].items():
                    for sym, size in symbols.items():
                        row = {"Symbol": sym, **{sec: 0 for sec in json_sections}}
                        row[section_name] = size
                        rows.append(row)
                df = pd.DataFrame(rows).fillna(0)
                if df.empty:
                    continue
                # ensure consistent column order
                df["Total"] = df[json_sections].sum(axis=1)
                df = df[["Symbol", *json_sections, "Total"]]
                df_sorted = df.sort_values(by="Total", ascending=False, kind="mergesort")
                sum_row = {"Symbol": "SUM", **{s: df_sorted[s].sum() for s in json_sections}, "Total": df_sorted["Total"].sum()}
                df_sorted = pd.concat([df_sorted, pd.DataFrame([sum_row])], ignore_index=True)
                md_lines.append(f"## {f['file']}")
                md_lines.append("")
                md_lines.append(df_sorted.to_markdown(index=False))
                md_lines.append("")
            if len(md_lines) == 4:  # nothing added
                md_lines.append("(no matching object files)")
        else:
            for f in json_data["files"]:
                row = {
                    "File": f["file"],
                    **{s: f["sections"].get(s, 0) for s in json_sections},
                    "Total": f.get("total", 0)
                }
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows).sort_values(by="Total", ascending=False)
                sum_row = {"File": "SUM", **{s: df[s].sum() for s in json_sections}, "Total": df["Total"].sum()}
                df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
                md_lines = [
                    "# Linker Map Summary",
                    "",
                    df.to_markdown(index=False)
                ]
            else:
                md_lines = ["# Linker Map Summary", "", "(no matching object files)"]

        with open(markdown_fname, 'w', encoding='utf-8') as mdfile:
            mdfile.write("\n".join(md_lines))
        if not quiet:
            print(f'Markdown summary written to {markdown_fname}')


if __name__ == '__main__':
    main()
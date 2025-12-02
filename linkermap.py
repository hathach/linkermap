__all__ = ["main", "analyze_map", "write_json", "write_markdown", "version_str"]
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
            # Normalize compiler-generated suffixes like .c.o / .c.obj back to .c
            if self.basepath.endswith('.c.o'):
                self.basepath = self.basepath[:-2]  # drop trailing .o
            elif self.basepath.endswith('.c.obj'):
                self.basepath = self.basepath[:-4]  # drop trailing .obj
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


def print_summary(json_data, verbose):
    section_list = json_data["sections"]
    files = json_data["files"]

    # Dynamic right-aligned width based on longest file/symbol name.
    name_candidates = [f["file"] for f in files]
    if verbose:
        for f in files:
            for sec_val in f["sections"].values():
                if isinstance(sec_val, dict):
                    name_candidates.extend(sec_val.keys())
    name_width = max(len(n) for n in name_candidates + ['SUM', 'File'])
    ffmt = '{:' + f'>{name_width}' + '} |'

    header = ffmt.format('File') + ''.join(map(sfmt.format, section_list)) + sfmt.format('Total')
    print(header)
    print('-'*len(header))

    sum_all = dict.fromkeys(section_list, 0)
    sum_all['total'] = 0

    for f in sorted(files, key=lambda x: x['total'], reverse=True):
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

    fd = open(map_file)
    sections = parseSections(fd)
    if sections is None:
        raise RuntimeError('start of memory config not found, did you invoke the compiler/linker with LANG=C?')

    sectionWhitelist = {'.text', '.data', '.bss', '.rodata', *extra_sections}
    whitelistedSections = list(filter(lambda x: x.section in sectionWhitelist, sections))

    files_out = []

    for s in whitelistedSections:
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
                entry = {"file": k, "path": group_list[0].path[0], "sections": {}, "total": 0}
                files_out.append(entry)
            entry.setdefault("sections", {}).setdefault(s.section, {})
            for symbol in sorted(group_list, reverse=True, key=lambda x: x.size):
                entry["sections"][s.section][symbol.children[0][1] if symbol.children else symbol.section] = symbol.size
                entry["total"] += symbol.size

    json_sections = list(map(lambda x: x.section, whitelistedSections))

    if not verbose:
        # collapse section dictionaries to totals
        for f in files_out:
            collapsed = {}
            for s in json_sections:
                if s in f["sections"]:
                    collapsed[s] = sum(f["sections"][s].values())
            f["sections"] = collapsed

    json_data = {
        "sections": json_sections,
        "files": files_out
    }

    return json_data


def write_json(json_data, path):
    with open(path, "w", encoding="utf-8") as outf:
        json.dump(json_data, outf, indent=2)


def write_markdown(json_data, path, verbose=False):
    json_sections = json_data["sections"]
    rows = []

    # Build mapfiles bullet list if present
    mapfiles_lines = []
    if "mapfiles" in json_data and json_data["mapfiles"]:
        mapfiles_lines = ["## Map Files", ""]
        for mf in json_data["mapfiles"]:
            mapfiles_lines.append(f"- {mf}")
        mapfiles_lines.append("")

    if verbose:
        md_lines = ["# Linker Map Summary", "", f"Sections included: {', '.join(json_sections)}", ""]
        md_lines.extend(mapfiles_lines)
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
        if len(md_lines) == 4 + len(mapfiles_lines):  # nothing added
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
                *mapfiles_lines,
                df.to_markdown(index=False)
            ]
        else:
            md_lines = ["# Linker Map Summary", "", *mapfiles_lines, "(no matching object files)"]

    with open(path, "w", encoding="utf-8") as mdfile:
        mdfile.write("\n".join(md_lines))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    map_file = args.map_file
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
        print_summary(json_data, verbose)

    if want_json:
        write_json(json_data, json_fname)
        if not quiet:
            print(f"JSON summary written to {json_fname}")

    if want_markdown:
        write_markdown(json_data, markdown_fname, verbose)
        if not quiet:
            print(f"Markdown summary written to {markdown_fname}")


if __name__ == '__main__':
    main()

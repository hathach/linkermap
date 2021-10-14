# vim: set fileencoding=utf8 :

import sys, re, os
from itertools import chain, groupby
import click
import linkermap.__version__

ffmt = '{:>50} |'
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


def print_file(verbose, header, symlist):
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
        print_file(verbose, header, sorted(finfo.items(), key=lambda x: sum(x[1].values()), reverse=True))

    # Sum
    print(ffmt.format('SUM') + ''.join(map(sfmt.format, sum_all.values())))


@click.version_option(linkermap.__version__.version_str)
@click.command()
@click.argument('map_file', required=True)
@click.option('-v', '--verbose', is_flag=True, help='Print symbols within file')

def main(map_file, verbose):
    fd = open(map_file)
    sections = parseSections (fd)
    if sections is None:
        print ('start of memory config not found, did you invoke the compiler/linker with LANG=C?')
        return

    sectionWhitelist = {'.text', '.data', '.bss', '.rodata'}
    whitelistedSections = list (filter (lambda x: x.section in sectionWhitelist, sections))
    #allObjects = list (chain (*map (lambda x: x.children, whitelistedSections)))
    #allFiles = list (set (map (lambda x: x.basepath, allObjects)))

    symbol_table = {}

    for s in whitelistedSections:
        objects = s.children
        for k, g in groupby(sorted(objects, key=lambda x: x.basepath), lambda x: x.basepath):
            symbol_table.setdefault(k, {})
            symbol_table[k].setdefault('total', 0)
            symbol_table[k].setdefault(s.section, {})
            for symbol in sorted(g, reverse=True, key=lambda x: x.size):
                symbol_table[k][s.section][symbol.children[0][1] if symbol.children else symbol.section] = symbol.size
                symbol_table[k]['total'] += symbol.size

    print_summary(verbose, list(map(lambda x: x.section, whitelistedSections)), symbol_table)

if __name__ == '__main__':
    main()

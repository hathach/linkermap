# linkermap

Analyze GNU ld’s linker map, based on [visual version](https://github.com/PromyLOPh/linkermapviz) .
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
$ linkermap --help
usage: linkermap [-h] [-v] [-s EXTRA_SECTIONS] [-j] [-f FILTER] [-m] [-V] map_file

positional arguments:
  map_file              Path to the linker map file to analyze.

options:
  -h, --help            Show this help message and exit
  -v, --verbose         Print symbols within file.
  -s EXTRA_SECTIONS, --section EXTRA_SECTIONS
                        Additional section name to include; repeat for multiple sections.
  -j, --json            Write JSON summary next to the map file (basename + .json).
  -f FILTER, --filter FILTER
                        Only include object files whose path contains this substring (can be repeated).
  -m, --markdown        Write Markdown table next to the map file (basename + .md).
  -V, --version         show program's version number and exit
```

### non-verbose

```bash
$ linkermap cdc_msc.elf.map
                                              File |   .text   .data    .bss   Total
------------------------------------------------------------------------------------
                                        msc_disk.o |    8192       1     313    8506
                                            usbd.o |      24     257    2820    3101
                                       dcd_nrf5x.o |       0     220    2678    2898
                                      msc_device.o |       0     576    2050    2626
                                            libc.a |    1072      33     696    1801
                                      cdc_device.o |       0     312    1044    1356
                                       tusb_fifo.o |       0       0    1142    1142
                                      nrfx_uarte.o |       0      36     800     836
                                    usbd_control.o |       0      84     516     600
                        gcc_startup_nrf52840_asm.o |       0       0     572     572
                                 system_nrf52840.o |       4       0     568     572
                                 usb_descriptors.o |      24      64     276     364
                                      nrfx_power.o |       0      13     330     343
                                          family.o |       8       4     274     286
                                            main.o |       4       5     188     197
                                            tusb.o |       0       0     176     176
                                            crt0.o |       0       0     124     124
                                        crtbegin.o |       8      25      64      97
                                       libc_nano.a |       0       0      96      96
                                            crtn.o |       0       0      16      16
                                            crti.o |       0       0       8       8
                                          crtend.o |       0       0       4       4
                                        libnosys.a |       0       0       4       4
                                               SUM |   25725    9336    1630   14759

```

### verbose

```
$ linkermap -v cdc_msc.elf.map
                                              File |   .text   .data    .bss   Total
------------------------------------------------------------------------------------
                                    usbd_control.o |       0      84     516     600
                                    -------------- |--------------------------------
                            usbd_control_xfer_cb   |       0       0     208     208
                                tud_control_xfer   |       0       0     112     112
                          .text._data_stage_xact   |       0       0      92      92
                             .bss._usbd_ctrl_buf   |       0      64       0      64
                              tud_control_status   |       0       0      48      48
                        usbd_control_set_request   |       0       0      28      28
                                 .bss._ctrl_xfer   |       0      20       0      20
                              usbd_control_reset   |       0       0      16      16
              usbd_control_set_complete_callback   |       0       0      12      12
------------------------------------------------------------------------------------
                        gcc_startup_nrf52840_asm.o |       0       0     572     572
                        -------------------------- |--------------------------------
                                    __isr_vector   |       0       0     512     512
                                   Reset_Handler   |       0       0      60      60
------------------------------------------------------------------------------------
                                 system_nrf52840.o |       4       0     568     572
                                 ----------------- |--------------------------------
                                      SystemInit   |       0       0     464     464
                           .text.nrf52_errata_36   |       0       0      60      60
                           .text.nrf52_errata_98   |       0       0      40      40
                                 SystemCoreClock   |       4       0       0       4
                                .rodata.CSWTCH.6   |       0       0       4       4
------------------------------------------------------------------------------------
                                 usb_descriptors.o |      24      64     276     364
                                 ----------------- |--------------------------------
                           desc_fs_configuration   |       0       0      98      98
                        tud_descriptor_string_cb   |       0       0      88      88
                                  .bss._desc_str   |       0      64       0      64
                                  .rodata.str1.1   |       0       0      54      54
                                 string_desc_arr   |      24       0       0      24
                                     desc_device   |       0       0      18      18
                        tud_descriptor_device_cb   |       0       0       8       8
                 tud_descriptor_configuration_cb   |       0       0       8       8
                    .rodata.__compound_literal.0   |       0       0       2       2
------------------------------------------------------------------------------------
```

### Markdown output

```bash
$ linkermap -m cdc_msc.elf.map

# Linker Map Summary
| File                      |   .text |   .rodata |   .data |   .bss |   Total |
|:--------------------------|--------:|----------:|--------:|-------:|--------:|
| msc_disk.o                |     300 |        13 |    8192 |      1 |    8506 |
| usbd.o                    |    2718 |        48 |      24 |    245 |    3035 |
| dcd_synopsys.o            |    2578 |         0 |       0 |    143 |    2721 |
| msc_device.o              |    2050 |         0 |       0 |    576 |    2626 |
| cdc_device.o              |    1044 |         0 |       0 |    312 |    1356 |
| stm32f4xx_hal_rcc.o       |    1316 |         0 |       0 |      0 |    1316 |
| tusb_fifo.o               |    1198 |         0 |       0 |      0 |    1198 |
| libgcc.a                  |     760 |         0 |       0 |      0 |     760 |
| family.o                  |     564 |         0 |       0 |     68 |     632 |
| usbd_control.o            |     516 |         0 |       0 |     84 |     600 |
| stm32f4xx_hal_gpio.o      |     414 |         0 |       0 |      0 |     414 |
| usb_descriptors.o         |     104 |       172 |      24 |     64 |     364 |
| stm32f4xx_hal_uart.o      |     326 |         0 |       0 |      0 |     326 |
| main.o                    |     188 |         0 |       4 |      5 |     197 |
| libc_nano.a               |     184 |         0 |       0 |      0 |     184 |
| tusb.o                    |     176 |         0 |       0 |      0 |     176 |
| stm32f4xx_hal_cortex.o    |     144 |         0 |       0 |      0 |     144 |
| stm32f4xx_hal.o           |      84 |         0 |       5 |      4 |      93 |
| startup_stm32f411xe_asm.o |      82 |         0 |       0 |      0 |      82 |
| system_stm32f4xx.o        |      28 |        24 |       4 |      0 |      56 |
| SUM                       |   14774 |       257 |    8253 |   1502 |   24786 |
```

**extra sections & JSON/Markdown output**

Enable JSON/Markdown outputs with flags; files are written beside the map using the map’s basename:

```bash
$ linkermap -s .fastcode -s .itcm -j -m build/firmware.map
```

The Markdown table includes a final `SUM` row with totals across all files. JSON and Markdown are saved as `<map_file>.json` / `<map_file>.md` in the same directory as the map file. Outputs follow the `-v/--verbose` flag: non-verbose shows per-file section totals; verbose lists each symbol per file with section names as columns plus a per-symbol Total column, sorted by size, and a per-file SUM row.
In verbose mode the Markdown file groups symbols under each file as separate tables for easier browsing. Use `-f/--filter` (repeatable) to include only object files whose path contains given substrings.

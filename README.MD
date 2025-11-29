# linkermap

Analyze GNU ld’s linker map, based on [visual version](https://github.com/PromyLOPh/linkermapviz) .
Works best with `-ffunction-sections -fdata-sections` and statically linked code (e.g. for bare-metal embedded systems).
Note: `-flto` merges code across .o files and make it difficult to analyze.

## Install

```bash
$ pip install .
$ linkermap --help
usage: linkermap [-h] [-v] [-s EXTRA_SECTIONS] [-o OUTPUT] [-V] map_file

positional arguments:
  map_file              Path to the linker map file to analyze.

options:
  -h, --help            show this help message and exit
  -v, --verbose         Print symbols within file.
  -s EXTRA_SECTIONS, --section EXTRA_SECTIONS
                        Additional section name to include; repeat for multiple sections.
  -o OUTPUT, --output OUTPUT
                        Write JSON summary to this file (default: <map_file basename>.json).
  -V, --version         show program's version number and exit
```

## Usages

**non-verbose**

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

**verbose**

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

**extra sections & JSON output**

Every run also writes a JSON summary next to the map file (e.g. `cdc_msc.elf.json`). Use `-s/--section` to add more whitelisted sections and `-o/--output` to choose the JSON path:

```bash
$ linkermap -s .fastcode -s .itcm -o reports/firmware.json build/firmware.map
```

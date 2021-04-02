project('cfish', 'c', default_options : 'c_std=c11')

cfish_src = [
  'src/benchmark.c', 'src/bitbase.c', 'src/bitboard.c', 'src/endgame.c', 'src/evaluate.c', 'src/material.c',
  'src/misc.c', 'src/movegen.c', 'src/movepick.c', 'src/nnue.c', 'src/pawns.c', 'src/position.c', 'src/psqt.c',
  'src/search.c', 'src/tbprobe.c', 'src/thread.c', 'src/timeman.c', 'src/tt.c', 'src/uci.c', 'src/ucioption.c',
  'src/numa.c', 'src/settings.c', 'src/polybook.c'
]

run_command('sed', '-i', '-e', 's/DefaultEvalFile.*/DefaultEvalFile "nn.bin"/', 'src/evaluate.h')

pext = false
avx = false
avx2 = false
avx512 = false
vnni = false

if get_option('arch').contains('avx')
  avx = true
endif

if get_option('arch').contains('avx2')
  avx2 = true
endif

if get_option('arch').contains('bmi2')
  avx2 = true
  pext = true
endif

if get_option('arch').contains('avx512')
  avx2 = true
  pext = true
  avx512 = true
endif

if get_option('arch').contains('vnni')
  avx2 = true
  pext = true
  vnni = true
endif

arguments = [ '-D_DEFAULT_SOURCE', '-D_BSD_SOURCE', '-DNDEBUG', '-DIS_64BIT' ]

if get_option('arch').contains('x86-64')
  arguments += [ '-msse3', '-mpopcnt', '-DUSE_POPCNT', '-DUSE_SSE41',
                 '-msse4.1', '-DUSE_SSSE3', '-mssse3', '-DUSE_SSE2',
                 '-msse2', '-msse', '-DUSE_SSE' ]
endif

if avx2
  arguments += [ '-DUSE_AVX2', '-mavx2' ]
elif avx
  arguments += [ '-mavx', '-DNNUE_SPARSE' ]
else
  arguments += [ '-DNNUE_SPARSE' ]
endif

if pext
  arguments += ['-DUSE_PEXT', '-msse4', '-mbmi2' ]
endif

if avx512
  arguments += [ '-DUSE_AVX512', '-mavx512f', '-mavx512bw' ]
endif

if vnni
  arguments += [ '-DUSE_VNNI', '-mavx512f', '-mavx512bw', '-mavx512vnni', '-mavx512dq', '-mavx512vl' ]
endif

cfish_dep = declare_dependency(
  sources: cfish_src,
  include_directories: 'src',
  compile_args: meson.get_compiler('c').get_supported_arguments(arguments),
)



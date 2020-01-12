del shaders_se.h

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_128 /DBLOCK_SIZE=128 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_256 /DBLOCK_SIZE=256 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_320 /DBLOCK_SIZE=320 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_384 /DBLOCK_SIZE=384 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt


dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_512 /DBLOCK_SIZE=512 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_640 /DBLOCK_SIZE=640 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_768 /DBLOCK_SIZE=768 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp32_se_1024 /DBLOCK_SIZE=1024 /Fh temp.txt WinogradTransformSE.hlsl
type temp.txt >> shaders_se.h
del temp.txt





dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_128 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=128 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_256 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=256 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_320 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=320 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_384 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=384 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_512 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=512 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_640 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=640 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_768 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=768 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

dxc /Tcs_6_2 /EOutputTransformSE -Vn g_output_transform_shader_fp16_se_1024 /DUSE_FP16_MATH=1 /DBLOCK_SIZE=1024 /Fh temp.txt WinogradTransformSE.hlsl  -enable-16bit-types
type temp.txt >> shaders_se.h
del temp.txt

msbuild "C:\projects\lc0\build\lc0.sln" /m /p:WholeProgramOptimization=PGInstrument /logger:"C:\Program Files\AppVeyor\BuildAgent\Appveyor.MSBuildLogger.dll"
IF ERRORLEVEL 1 EXIT
cd build
IF %NAME%==cpu-openblas copy C:\cache\OpenBLAS\dist64\bin\libopenblas.dll
IF %NAME%==cpu-dnnl copy C:\cache\dnnl_win_1.1.1_cpu_vcomp\bin\dnnl.dll
IF %OPENCL%==true copy C:\cache\opencl-nug.0.777.77\build\native\bin\OpenCL.dll
IF %CUDA%==true copy "%CUDA_PATH%"\bin\*.dll
IF %CUDA%==true copy %PKG_FOLDER%\cuda\bin\cudnn64_7.dll
REM We don't care if this fails (e.g. for dx12), the rest of the build will work.
lc0 benchmark --num-positions=1 --weights=c:\cache\591226.pb.gz --backend=random --movetime=10000
cd ..
msbuild "C:\projects\lc0\build\lc0.sln" /m /p:WholeProgramOptimization=PGOptimize /p:DebugInformationFormat=ProgramDatabase /logger:"C:\Program Files\AppVeyor\BuildAgent\Appveyor.MSBuildLogger.dll"
IF %APPVEYOR_REPO_TAG%==false IF EXIST build\lc0.pdb del build\lc0.pdb

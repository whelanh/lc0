git clone https://github.com/lealgo/chessenginesupport-androidlib.git --branch lc0 --single-branch oex
cd oex
git checkout 87e0168310c72dd059d9f574f4434f7f32de1214
cd ..
perl -e 'printf "%sLc0!", pack("V", -s "c:/cache/591226.pb.gz")' >tail.bin
copy /y /b arm64-v8a\lc0+c:\cache\591226.pb.gz+tail.bin oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\arm64-v8a\liblc0.so
copy /y /b armeabi-v7a\lc0+c:\cache\591226.pb.gz+tail.bin oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\armeabi-v7a\liblc0.so
set ANDROID_HOME=C:\android-sdk-windows
appveyor DownloadFile https://dl.google.com/android/repository/sdk-tools-windows-3859397.zip
7z x sdk-tools-windows-3859397.zip -oC:\android-sdk-windows > nul
yes | C:\android-sdk-windows\tools\bin\sdkmanager.bat --licenses
cd oex\LeelaChessEngine
sed -i "s/0.22/%APPVEYOR_REPO_TAG_NAME%/g" leelaChessEngine/src/main/AndroidManifest.xml
call gradlew.bat assemble
copy leelaChessEngine\build\outputs\apk\debug\leelaChessEngine-debug.apk ..\..\lc0-%APPVEYOR_REPO_TAG_NAME%-android.apk

git clone https://github.com/lealgo/chessenginesupport-androidlib.git --branch lc0 --single-branch oex
cd oex
git checkout ca00901abb53c5e56098b446c15e1d9236dd48f5
cd ..
7z a -mx=0 embed.zip c:\cache\591226.pb.gz
copy /y /b arm64-v8a\lc0+embed.zip oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\arm64-v8a\liblc0.so
copy /y /b armeabi-v7a\lc0+embed.zip oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\armeabi-v7a\liblc0.so
set ANDROID_HOME=C:\android-sdk-windows
appveyor DownloadFile https://dl.google.com/android/repository/sdk-tools-windows-3859397.zip
7z x sdk-tools-windows-3859397.zip -oC:\android-sdk-windows > nul
yes | C:\android-sdk-windows\tools\bin\sdkmanager.bat --licenses
cd oex\LeelaChessEngine
call gradlew.bat assemble
copy leelaChessEngine\build\outputs\apk\debug\leelaChessEngine-debug.apk ..\..\lc0-%APPVEYOR_REPO_TAG_NAME%-android.apk

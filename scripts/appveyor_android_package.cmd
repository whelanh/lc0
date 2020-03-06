git clone https://github.com/lealgo/chessenginesupport-androidlib.git --branch lc0 --single-branch oex
del oex\LeelaChessEngine\leelaChessEngine\src\main\assets\networks\embed.zip
7z a -mx=0 oex\LeelaChessEngine\leelaChessEngine\src\main\assets\networks\embed.zip c:\cache\591226.pb.gz
copy /y arm64-v8a\lc0 oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\arm64-v8a\liblc0.so
copy /y armeabi-v7a\lc0 oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\armeabi-v7a\liblc0.so
set ANDROID_HOME=C:\android-sdk-windows
appveyor DownloadFile https://dl.google.com/android/repository/sdk-tools-windows-3859397.zip
7z x sdk-tools-windows-3859397.zip -oC:\android-sdk-windows > nul
yes | C:\android-sdk-windows\tools\bin\sdkmanager.bat --licenses
cd oex\LeelaChessEngine
gradlew.bat assemble
copy leelaChessEngine\build\outputs\apk\debug\leelaChessEngine-debug.apk ..\..\lc0-%APPVEYOR_REPO_TAG_NAME%-android.apk

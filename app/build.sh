#!/bin/bash
# Build + sign hexanima.apk without gradle (SDK in ~/anima-npu/sdk/android, JDK from the temurin docker image), then install on the phone.
cd "$(dirname "$0")"
cp ../phone/anima.sh assets/anima.sh; cp ../phone/moss.sh assets/moss.sh; cp ../phone/qwen.sh assets/qwen.sh; cp ../phone/mage.sh assets/mage.sh
SDK=~/anima-npu/sdk/android
docker run --rm -v $SDK:/sdk -v $PWD:/app eclipse-temurin:17-jdk bash -c '
set -e; BT=/sdk/build-tools/34.0.0; AJ=/sdk/platforms/android-34/android.jar; cd /app; rm -rf out; mkdir -p out/obj out/cls
$BT/aapt2 compile --dir res -o out/res.zip
$BT/aapt2 link -o out/base.apk -I $AJ -A assets --manifest AndroidManifest.xml --min-sdk-version 29 --target-sdk-version 34 out/res.zip
javac -encoding UTF-8 --release 11 -classpath $AJ -d out/cls src/com/hexanima/npu/*.java
$BT/d8 --lib $AJ --min-api 29 --output out out/cls/com/hexanima/npu/*.class
cd out && cp base.apk unsigned.apk && $BT/aapt add unsigned.apk classes.dex >/dev/null && cd ..
[ -d jnilibs ] && (cd jnilibs && $BT/aapt add ../out/unsigned.apk $(find lib -type f) >/dev/null)
[ -f /sdk/debug.keystore ] || keytool -genkeypair -keystore /sdk/debug.keystore -storepass android -keypass android -alias k -keyalg RSA -keysize 2048 -validity 10000 -dname CN=anima
$BT/zipalign -f 4 out/unsigned.apk out/aligned.apk
$BT/apksigner sign --ks /sdk/debug.keystore --ks-pass pass:android --out hexanima.apk out/aligned.apk
rm -rf out; ls -l hexanima.apk' || exit 1
adb install -r hexanima.apk

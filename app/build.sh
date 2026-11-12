#!/bin/bash
# Build + sign hexanima.apk without gradle (SDK in ~/anima-npu/sdk/android, JDK from the temurin docker image), then install on the phone.
cd "$(dirname "$0")"
cp ../phone/anima.sh assets/anima.sh; cp ../phone/moss.sh assets/moss.sh; cp ../phone/qwen.sh assets/qwen.sh; cp ../phone/mage.sh assets/mage.sh; cp ../phone/h3.sh assets/h3.sh
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
# the install must never cost the user's data: refuse without a device, and copy the in-app gallery (pictures + metadata) to /sdcard first
adb get-state >/dev/null 2>&1 || { echo "no device: not installing"; exit 1; }
if adb shell "su 0 sh -c 'test -d /data/data/com.hexanima.npu/files/gallery'" 2>/dev/null; then
  adb shell "su 0 sh -c 'mkdir -p /sdcard/HexAnimaBackup && cp -r /data/data/com.hexanima.npu/files/gallery /sdcard/HexAnimaBackup/ 2>/dev/null; cp /data/data/com.hexanima.npu/shared_prefs/*.xml /sdcard/HexAnimaBackup/ 2>/dev/null; true'"
fi
adb install -r hexanima.apk

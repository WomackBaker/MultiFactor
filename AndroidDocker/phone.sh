#!/bin/bash

# Start the Python script
python3 /opt/refer.py > /opt/flask.log 2>&1 &

# Define an array of available Pixel device names
PIXEL_DEVICES=("pixel_2" "pixel_2_xl" "pixel_3" "pixel_3_xl" "pixel_4" "pixel_4_xl")

# Randomly select a Pixel device
RANDOM_DEVICE=${PIXEL_DEVICES[$RANDOM % ${#PIXEL_DEVICES[@]}]}

echo "Selected device: $RANDOM_DEVICE"
# Create the AVD for the selected device
echo "no" | $ANDROID_SDK_ROOT/cmdline-tools/bin/avdmanager create avd -n $RANDOM_DEVICE -k "system-images;android-30;google_apis;x86_64" -d $RANDOM_DEVICE

# Modify config.ini to remove android-sdk/ from image.sysdir.1
sed -i 's|android-sdk/||' /root/.android/avd/${RANDOM_DEVICE}.avd/config.ini

# Start the emulator for the selected AVD
$ANDROID_SDK_ROOT/emulator/emulator -avd $RANDOM_DEVICE -no-audio -no-window -gpu swiftshader_indirect -verbose &

# Wait for the emulator to start
sleep 60

adb -s emulator-5554 root

adb -s emulator-5554 shell settings put secure location_providers_allowed +gps,network

# Set the GPS location
LATITUDE=$(awk -v min=-90 -v max=90 'BEGIN{srand(); print min+rand()*(max-min+1)}')
LONGITUDE=$(awk -v min=-180 -v max=180 'BEGIN{srand(); print min+rand()*(max-min+1)}')

echo "Setting GPS location to Latitude: $LATITUDE, Longitude: $LONGITUDE"

adb -s emulator-5554 emu geo fix $LONGITUDE $LATITUDE

# Trigger a location update
adb -s emulator-5554 shell am broadcast -a android.intent.action.PROVIDER_CHANGED -c android.intent.category.DEFAULT

# Connect to the emulator and install the APK
adb -s emulator-5554 install /opt/multifactorapp.apk
adb -s emulator-5554 shell pm grant com.example.multifactorapp android.permission.ACCESS_FINE_LOCATION
adb -s emulator-5554 shell pm grant com.example.multifactorapp android.permission.ACCESS_COARSE_LOCATION
adb -s emulator-5554 shell pm grant com.example.multifactorapp android.permission.READ_PHONE_STATE
adb -s emulator-5554 shell pm grant com.example.multifactorapp android.permission.RECORD_AUDIO

# Start the main activity
adb -s emulator-5554 shell am start -n com.example.multifactorapp/.MainActivity
sleep 45

# Extract the UUID from the shared preferences XML
UUID=$(adb -s emulator-5554 shell "run-as com.example.multifactorapp cat /data/data/com.example.multifactorapp/shared_prefs/AppPrefs.xml" | grep -oP '(?<=<string name="uuid">).*(?=</string>)')

echo "Extracted UUID: $UUID"

# Loop to start GetDataActivity with the UUID
while true
do
  adb -s emulator-5554 shell am start -n com.example.multifactorapp/.GetDataActivity --es username "$UUID"
  sleep 10
done

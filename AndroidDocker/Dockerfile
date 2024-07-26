# Baker Womack
# docker build -t android .
# docker run --rm -it --device /dev/kvm --privileged -p 30083:30083 -p 5555:5555 android
# docker tag android bwomack/android:latest
# docker push bwomack/android:latest

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    wget \
    unzip \
    libgl1-mesa-dev \
    libqt5widgets5 \
    adb \
    qemu-kvm \
    python3 \
    python3-pip \
    python3-requests \
    python3-flask \
    && apt-get clean

RUN pip3 install requests flask

# Set environment variables
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_SDK_ROOT/tools:$ANDROID_SDK_ROOT/tools/bin:$ANDROID_SDK_ROOT/platform-tools

# Download and install Android SDK
RUN mkdir -p $ANDROID_SDK_ROOT \
    && cd $ANDROID_SDK_ROOT \
    && wget https://dl.google.com/android/repository/commandlinetools-linux-6858069_latest.zip -O cmdline-tools.zip \
    && unzip cmdline-tools.zip -d . \
    && rm cmdline-tools.zip

# Install Android SDK components
RUN yes | $ANDROID_SDK_ROOT/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_SDK_ROOT --licenses \
    && $ANDROID_SDK_ROOT/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_SDK_ROOT "platform-tools" "emulator" "platforms;android-30" "system-images;android-30;google_apis;x86_64"

# Create a script to randomly select and configure an AVD
COPY phone.sh /usr/local/bin/phone.sh
RUN chmod +x /usr/local/bin/phone.sh

# Copy the APK file into the container
COPY "." /opt/

# Expose ports for ADB and Emulator
EXPOSE 5554 5555

# Start the emulator, install the APK, and run the app using the random AVD
CMD /usr/local/bin/phone.sh && tail -f /dev/null

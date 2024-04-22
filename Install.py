import subprocess

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        if result.stderr:
            print("Error:", result.stderr)
        else:
            print("Output:", result.stdout)
    except Exception as e:
        print("An error occurred:", e)

def main():
    # Starting BlueStacks with a specific VM
    execute_command('BlueStacks.exe -runvm "Android7_1" -vm-name "MyPhone1"')

    # Installing the APK not available on the Google Play Store
    execute_command("adb -s MyPhone1 install path/to/your/com.example.multifactor.apk")

    # Running the app
    execute_command("adb -s MyPhone1 shell monkey -p com.example.multifactor -c android.intent.category.LAUNCHER 1")

if __name__ == "__main__":
    main()

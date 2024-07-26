# Path to the text file containing the AVD names
$avdFilePath = ".\vms.txt"

# Function to start an emulator
function Start-Emulator {
    param (
        [string]$avdName
    )
    Write-Host "Starting emulator for AVD: $avdName"
    Start-Process -NoNewWindow -FilePath "emulator" -ArgumentList "-avd $avdName"
}

# Function to run an app
function Run-App {
    param (
        [string]$packageName,
        [string]$mainActivity,
        [string]$apkPath
    )

    # Check if the app is installed
    $appInstalled = & adb shell pm list packages | Where-Object { $_ -match $packageName }

    # If the app is not installed, install it
    if (-not $appInstalled) {
        Write-Host "App $packageName is not installed. Installing..."
        & adb install $apkPath
    }

    # Run the app
    Write-Host "Running app: $packageName"
    & adb shell am start -S -n "$packageName/$mainActivity"
}

# Function to check if the emulator is booted
function Wait-For-Device {
    Write-Host "Waiting for the emulator to boot..."
    & adb wait-for-device
    Write-Host "Emulator booted successfully!"
}

# Read the AVD names from the file
if (Test-Path $avdFilePath) {
    $avds = Get-Content -Path $avdFilePath
} else {
    Write-Host "The file at path '$avdFilePath' does not exist."
    exit 1
}

# Main script logic
foreach ($avd in $avds) {
    Start-Emulator -avdName $avd
    Wait-For-Device
    Run-App -packageName "com.example.multifactorapp" -mainActivity ".MainActivity" -apkPath ".\MultifactorApp\app\build\outputs\apk\debug\app-debug.apk"
}
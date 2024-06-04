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
        [string]$mainActivity
    )
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
    Run-App -packageName "com.example.multifactorapp" -mainActivity ".MainActivity"
}
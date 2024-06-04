# Number of AVDs to create
$numberOfAvds = 2

# List of device types
$deviceTypes = "pixel", "pixel_xl", "pixel_2", "pixel_2_xl", "pixel_3", "pixel_3_xl", "pixel_3a", "pixel_3a_xl", "pixel_4", "pixel_4_xl", "pixel_4a", "pixel_5"

# Android system image package
$package = "system-images;android-34;google_apis;x86_64"

# Function to create an emulator
function Create-Emulator {
    param (
        [string]$avdName,
        [string]$device,
        [string]$package
    )
    Write-Host "Creating emulator for AVD: $avdName, Device: $device"
    & avdmanager create avd -n $avdName -d $device -k $package
}

# Main script logic
for ($i = 1; $i -le $numberOfAvds; $i++) {
    # Generate a random device type
    $device = $deviceTypes | Get-Random

    # Create the AVD
    Create-Emulator -avdName "user$i" -device $device -package $package

    # Write the AVD name to the file
    "user$i" | Out-File -Append -Encoding ASCII -FilePath "vms.txt"
}
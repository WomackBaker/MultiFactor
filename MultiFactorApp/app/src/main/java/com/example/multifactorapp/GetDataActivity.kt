import android.Manifest
import android.app.Activity
import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Bundle
import android.text.format.Formatter
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationServices
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class DataSenderActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_data_sender)

        // Retrieve the username from the intent
        val username = intent.getStringExtra("username") ?: "default_user"

        // Send the device information
        sendDeviceInfo(this, username)
    }

    /**
     * Sends device information including location to a server.
     * @param context Context used to access system services.
     * @param username The username associated with the data being sent.
     */
    private fun sendDeviceInfo(context: Context, username: String) {
        // Access the Wi-Fi system service to get network details
        val wifiManager =
            context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val ipAddress = wifiManager.connectionInfo.ipAddress
        val ipString = Formatter.formatIpAddress(ipAddress)

        // Access the Activity system service to get memory info
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val availableMemory = memoryInfo.availMem

        // Additional device info
        val currentTime = System.currentTimeMillis()
        val dateFormatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        dateFormatter.timeZone = TimeZone.getTimeZone("America/New_York")  // Set the timezone
        val formattedTime = dateFormatter.format(Date(currentTime))
        val rssi = wifiManager.connectionInfo.rssi
        val timeZone = TimeZone.getDefault().id
        val numberOfProcessors = Runtime.getRuntime().availableProcessors()
        val batteryManager =
            context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
        val batteryPercent =
            batteryManager.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val vendor = Build.MANUFACTURER
        val model = Build.MODEL
        val systemPerformance = Runtime.getRuntime().availableProcessors()
        val cpuClass = Build.HARDWARE
        val accel = Sensor.TYPE_ACCELEROMETER
        val gyro = Sensor.TYPE_GYROSCOPE
        val magnet = Sensor.TYPE_MAGNETIC_FIELD
        val metrics = Resources.getSystem().displayMetrics
        val screenWidth = metrics.widthPixels
        val screenHeight = metrics.heightPixels
        val screenDensity = metrics.densityDpi
        val hasTouchScreen = context.packageManager.hasSystemFeature(PackageManager.FEATURE_TOUCHSCREEN)
        val wifiManager = context.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val wifiList = wifiManager.scanResults.map { it.SSID }
        val hasCamera = context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA)
        val hasFrontCamera = context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_FRONT)
        val hasMicrophone = context.packageManager.hasSystemFeature(PackageManager.FEATURE_MICROPHONE)
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val hasTemperatureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE) != null

        // Assemble the data into a JSON object
        val jsonObject = JSONObject().apply {
            put("user", username)
            put("ipString", ipString)
            put("availableMemory", availableMemory)
            put("currentTime", formattedTime)
            put("rssi", rssi)
            put("timezone", timeZone)
            put("Processors", numberOfProcessors)
            put("Battery", batteryPercent)
            put("Vendor", vendor)
            put("Model", model)
            put("systemPerformance", systemPerformance)
            put("cpu", cpuClass)
            put("accel", accel)
            put("gyro", gyro)
            put("magnet", magnet)
            put("WifiList", wifiList)
            put("screenWidth", screenWidth)
            put("screenLength", screenHeight)
            put("screenDensity", screenDensity)
            put("hasTouchScreen", hasTouchScreen)
            put("hasCamera", hasCamera)
            put("hasFrontCamera", hasFrontCamera)
            put("hasMicrophone", hasMicrophone)
            put("hasTemperatureSensor", hasTemperatureSensor)
        }

        // Check for location permissions before accessing location
        if (ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                context as Activity,
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ),
                0
            )
        } else {
            // Get last known location and update the JSON object with location data
            val fusedLocationClient = LocationServices.getFusedLocationProviderClient(context)

            fusedLocationClient.lastLocation.addOnSuccessListener { location ->
                if (location != null) {
                    jsonObject.put("latitude", location.latitude)
                    jsonObject.put("longitude", location.longitude)
                    // Send the data to the server
                    sendHttpRequest(context, jsonObject)
                } else {
                    Toast.makeText(context, "Location was not found.", Toast.LENGTH_LONG).show()
                }
            }.addOnFailureListener { e ->
                Toast.makeText(context, "Failed to get location: ${e.message}", Toast.LENGTH_LONG)
                    .show()
            }
        }
    }

    private fun sendHttpRequest(context: Context, jsonObject: JSONObject) {
        val mediaType = "application/json; charset=utf-8".toMediaType()
        val requestBody = RequestBody.create(mediaType, jsonObject.toString())

        // Build and execute the HTTP request
        val request = Request.Builder()
            .url("http://10.0.2.2:30083/data")
            .post(requestBody)
            .build()

        val client = OkHttpClient()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // Error handling for failed HTTP request
                Log.d("DataSenderActivity", "Failed to connect: ${e.message}")
                runOnUiThread {
                    Toast.makeText(
                        context,
                        "Failed to fetch data: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!it.isSuccessful) {
                        Log.d("DataSenderActivity", "Unexpected response code: ${response.code}")
                        throw IOException("Unexpected code $response")
                    }

                    Log.d("DataSenderActivity", "Data sent successfully")
                    // Notify success on the UI thread
                    runOnUiThread {
                        Toast.makeText(
                            context,
                            "Data sent successfully",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        })
    }
}

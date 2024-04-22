import android.app.Activity
import android.content.Context
import android.location.LocationManager
import android.net.wifi.WifiManager
import android.text.format.Formatter
import android.app.ActivityManager
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject
import java.io.IOException
import android.Manifest
import android.hardware.Sensor
import android.os.Build
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import java.util.TimeZone

object DataSender {

    fun fetchSampleData(context: Context, apiUrl: String) {
        val client = OkHttpClient()
        val request = Request.Builder()
            .url(apiUrl)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // Handle the error
                (context as Activity).runOnUiThread {
                    Toast.makeText(context, "Failed to fetch data: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!it.isSuccessful) {
                        (context as Activity).runOnUiThread {
                            Toast.makeText(context, "Failed to fetch data: ${response.message}", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        val responseData = it.body?.string()
                    }
                }
            }
        })
    }
    fun sendDeviceInfo(context: Context, username: String) {
        fun fetchSampleData(context: Context, apiUrl: String) {
                val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
                val ipAddress = wifiManager.connectionInfo.ipAddress
                val ipString = Formatter.formatIpAddress(ipAddress)
                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memoryInfo = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memoryInfo)
                val availableMemory = memoryInfo.availMem
                val currentTime = System.currentTimeMillis()
                val rssi = wifiManager.connectionInfo.rssi
                val timeZone = TimeZone.getDefault().id
                val numberOfProcessors = Runtime.getRuntime().availableProcessors()
                val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
                val batteryPercent = batteryManager.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
                val vendor = Build.MANUFACTURER
                val model = Build.MODEL
                val systemPerformance = Runtime.getRuntime().availableProcessors()
                val cpuClass = Build.HARDWARE
                val accel = Sensor.TYPE_ACCELEROMETER
                val gyro = Sensor.TYPE_GYROSCOPE
                val magnet = Sensor.TYPE_MAGNETIC_FIELD

                // Create JSON object
                val jsonObject = JSONObject().apply {
                    put("user", username)
                    put("ipString", ipString)
                    put("availableMemory", availableMemory)
                    put("currentTime", currentTime)
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
                }

                val mediaType = "application/json; charset=utf-8".toMediaType()
                val requestBody = RequestBody.create(mediaType, jsonObject.toString())

                val request = Request.Builder()
                    .url("http://10.0.2.2:30081/data")
                    .post(requestBody)
                    .build()

                val client = OkHttpClient()

                client.newCall(request).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        // Handle the error
                        (context as Activity).runOnUiThread {
                            Toast.makeText(context, "Failed to fetch data: ${e.message}", Toast.LENGTH_SHORT).show()
                        }
                    }
                    override fun onResponse(call: Call, response: Response) {
                        response.use {
                            if (!it.isSuccessful) throw IOException("Unexpected code $response")

                            (context as Activity).runOnUiThread {
                                Toast.makeText(context, "Data sent successfully", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                })
            }
        }
    }
}

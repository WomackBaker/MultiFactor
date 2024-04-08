package com.example.multifactorapp

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

object DataSender {
    fun sendDeviceInfo(context: Context) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val ipAddress = wifiManager.connectionInfo.ipAddress
        val ipString = Formatter.formatIpAddress(ipAddress)
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val availableMemory = memoryInfo.availMem
        val currentTime = System.currentTimeMillis()

        // Create JSON object
        val jsonObject = JSONObject().apply {
            put("user", "Baker")
            put("latitude", location?.latitude ?: 9)
            put("longitude", location?.longitude ?: 9)
            put("ipString", ipString)
            put("availableMemory", availableMemory)
            put("currentTime", currentTime)
        }

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val requestBody = RequestBody.create(mediaType, jsonObject.toString())

        val request = Request.Builder()
            .url("http://10.0.2.2:8080/get-data")
            .post(requestBody)
            .build()

        val client = OkHttpClient()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                (context as Activity).runOnUiThread {
                    Toast.makeText(context, "Network error or cannot reach the server", Toast.LENGTH_SHORT).show()
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

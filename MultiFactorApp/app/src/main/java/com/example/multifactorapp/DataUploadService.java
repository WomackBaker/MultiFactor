package com.example.multifactorapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.app.IntentService;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.text.format.Formatter;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationServices;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class DataUploadService extends IntentService {

    private static final String TAG = "DataUploadService";
    private FusedLocationProviderClient fusedLocationClient;

    public DataUploadService() {
        super(DataUploadService.class.getSimpleName());
    }

    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        if (intent != null) {
            String username = intent.getStringExtra("username");
            if (username == null) {
                username = "default_user";
            }

            // Check for location permissions
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                // Permissions are not granted, stop the service and notify the user
                Log.d(TAG, "Location permissions are not granted. Stopping service.");
                stopSelf();
                return;
            }

            // Initialize the fused location client
            fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);

            // Send the device information
            sendDeviceInfo(this, username);
        }
    }

    @SuppressLint("MissingPermission")
    private void sendDeviceInfo(Context context, String username) {
        // Access the Wi-Fi system service to get network details
        WifiManager wifiManager = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        int ipAddress = wifiManager.getConnectionInfo().getIpAddress();
        String ipString = Formatter.formatIpAddress(ipAddress);

        // Access the Activity system service to get memory info
        ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memoryInfo);
        long availableMemory = memoryInfo.availMem;

        // Additional device info
        long currentTime = System.currentTimeMillis();
        SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
        dateFormatter.setTimeZone(TimeZone.getTimeZone("America/New_York")); // Set the timezone
        String formattedTime = dateFormatter.format(new Date(currentTime));
        int rssi = wifiManager.getConnectionInfo().getRssi();
        String timeZone = TimeZone.getDefault().getID();
        int numberOfProcessors = Runtime.getRuntime().availableProcessors();
        android.os.BatteryManager batteryManager = (android.os.BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        int batteryPercent = batteryManager.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY);
        String vendor = Build.MANUFACTURER;
        String model = Build.MODEL;
        String cpuClass = Build.HARDWARE;
        int accel = Sensor.TYPE_ACCELEROMETER;
        int gyro = Sensor.TYPE_GYROSCOPE;
        int magnet = Sensor.TYPE_MAGNETIC_FIELD;
        android.util.DisplayMetrics metrics = Resources.getSystem().getDisplayMetrics();
        int screenWidth = metrics.widthPixels;
        int screenHeight = metrics.heightPixels;
        int screenDensity = metrics.densityDpi;
        boolean hasTouchScreen = context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_TOUCHSCREEN);
        boolean hasCamera = context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA);
        boolean hasFrontCamera = context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_FRONT);
        boolean hasMicrophone = context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_MICROPHONE);
        SensorManager sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        boolean hasTemperatureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE) != null;

        // Assemble the data into a JSON object
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("user", username);
            jsonObject.put("ipString", ipString);
            jsonObject.put("availableMemory", availableMemory);
            jsonObject.put("currentTime", formattedTime);
            jsonObject.put("rssi", rssi);
            jsonObject.put("timezone", timeZone);
            jsonObject.put("Processors", numberOfProcessors);
            jsonObject.put("Battery", batteryPercent);
            jsonObject.put("Vendor", vendor);
            jsonObject.put("Model", model);
            jsonObject.put("cpu", cpuClass);
            jsonObject.put("accel", accel);
            jsonObject.put("gyro", gyro);
            jsonObject.put("magnet", magnet);
            jsonObject.put("screenWidth", screenWidth);
            jsonObject.put("screenLength", screenHeight);
            jsonObject.put("screenDensity", screenDensity);
            jsonObject.put("hasTouchScreen", hasTouchScreen);
            jsonObject.put("hasCamera", hasCamera);
            jsonObject.put("hasFrontCamera", hasFrontCamera);
            jsonObject.put("hasMicrophone", hasMicrophone);
            jsonObject.put("hasTemperatureSensor", hasTemperatureSensor);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Get the current location
        fusedLocationClient.getCurrentLocation(LocationRequest.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener(location -> {
                    if (location != null) {
                        try {
                            jsonObject.put("latitude", location.getLatitude());
                            jsonObject.put("longitude", location.getLongitude());
                            // Send the data to the server
                            sendHttpRequest(context, jsonObject);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    } else {
                        Toast.makeText(context, "Location was not found.", Toast.LENGTH_LONG).show();
                        try {
                            jsonObject.put("longitude", 0);
                            jsonObject.put("latitude", 0);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                        // Send the data to the server
                        sendHttpRequest(context, jsonObject);
                    }
                })
                .addOnFailureListener(e -> {
                    Toast.makeText(context, "Failed to get location: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    try {
                        jsonObject.put("longitude", 0);
                        jsonObject.put("latitude", 0);
                        // Send the data to the server
                        sendHttpRequest(context, jsonObject);
                    } catch (JSONException jsonException) {
                        jsonException.printStackTrace();
                    }
                });
    }

    private void sendHttpRequest(Context context, JSONObject jsonObject) {
        MediaType mediaType = MediaType.get("application/json; charset=utf-8");
        RequestBody requestBody = RequestBody.create(mediaType, jsonObject.toString());

        // Build and execute the HTTP request
        Request request = new Request.Builder()
                .url("http://10.0.2.2:30083/data")
                .post(requestBody)
                .build();

        OkHttpClient client = new OkHttpClient();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // Error handling for failed HTTP request
                Log.d(TAG, "Data did not send successfully");
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    throw new IOException("Unexpected code " + response);
                }

                Log.d(TAG, "Data sent successfully");
                // Notify success on the UI thread
            }
        });
    }
}

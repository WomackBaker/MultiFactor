package com.example.multifactorapp

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.multifactorapp.ui.theme.MultifactorAppTheme
import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.remember
import androidx.core.content.ContextCompat
import java.util.UUID

/**
 * Main activity for the multifactor authentication app.
 * Handles user interactions and permissions related to device data transmission.
 */
class MainActivity : ComponentActivity() {
    companion object {
        internal const val PREFS_FILE = "AppPrefs"
        const val UUID_KEY = "uuid"
    }

    private lateinit var sharedPreferences: SharedPreferences

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions.entries.all { it.value }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sharedPreferences = this.getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)

        setContent {
            MultifactorAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ButtonsScreen(getOrCreateUUID())
                }
            }
        }
        // Check if location permissions are already granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            // Permissions are granted, proceed to send device info

        } else {
            // Not granted, request permissions
            requestPermissionLauncher.launch(
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            )
        }
    }

    private fun getOrCreateUUID(): String {
        // Check if UUID exists
        var uuid = sharedPreferences.getString(UUID_KEY, null)
        if (uuid == null) {
            // Generate a new UUID
            uuid = UUID.randomUUID().toString()
            // Save the new UUID to SharedPreferences
            sharedPreferences.edit().putString(UUID_KEY, uuid).apply()
        }
        return uuid
    }
}


@Composable
fun ButtonsScreen(getUUID: String) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Random Button
        Button(
            onClick = { /* TODO: Implement action */ },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .height(60.dp)
        ) {
            Text("Random")
        }

        // Face Button
        val facecontext = LocalContext.current
        Button(
            onClick = {facecontext.getActivity()?.let {
                ImagePickerActivity.start(it)
            }},
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .height(60.dp)
        ) {
            Text("Facial Recognition")
        }
        // Voice Button
        val Voicecontext = LocalContext.current
        Button(
            onClick = {Voicecontext.getActivity()?.let {
                VoicePickerActivity.start(it)

            }},
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .height(60.dp)
        ) {
            Text("Voice Recognition")
        }
        /**
         * Retrieves the UUID stored in SharedPreferences.
         */
        fun getUUID(context: Context): String {
            val sharedPreferences = context.getSharedPreferences(MainActivity.PREFS_FILE, Context.MODE_PRIVATE)
            return sharedPreferences.getString(MainActivity.UUID_KEY, "") ?: ""
        }

        // SMS Button
        // Button to send device info as an SMS-like feature
        val smscontext = LocalContext.current
        val uuid = getUUID(smscontext)
        Button(
            onClick = {val intent = Intent(smscontext, GetDataActivity::class.java).apply { putExtra("username", uuid)
                }
                Log.d("UUIDLogger", "Generated UUID: $uuid")
                smscontext.startActivity(intent)
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .height(60.dp)
        ) {
            Text("SMS")
        }

        // Pass Button
        Button(
            onClick = { /* TODO: Implement action */ },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .height(60.dp)
        ) {
            Text("Password")
        }
    }
}
/**
 * Extension function to get Activity from Context.
 */
fun Context.getActivity(): Activity? {
            var context = this
            while (context is ContextWrapper) {
                if (context is Activity) {
                    return context
                }
                context = context.baseContext
            }
            return null
        }

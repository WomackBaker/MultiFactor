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
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions.entries.all { it.value }
        if (granted) {
            // All requested permissions are granted
           // DataSender.sendDeviceInfo(this)
        } else {
            // Permissions denied, handle accordingly
            Toast.makeText(this, "Location permission is required to send device info.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check if location permissions are already granted
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

            // Not granted, request permissions
            requestPermissionLauncher.launch(
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            )
        } else {
            // Permissions are already granted
            DataSender.sendDeviceInfo(this)
        }

        setContent {
            MultifactorAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ButtonsScreen()
                }
            }
        }
    }
}


@Composable
fun ButtonsScreen() {
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
        val Facecontext = LocalContext.current
        Button(
            onClick = {Facecontext.getActivity()?.let {
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

        // SMS Button
        val smscontext = LocalContext.current
        Button(
            onClick = { DataSender.sendDeviceInfo(smscontext)},
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

package com.example.multifactorapp

import android.app.Activity
import android.content.Context
import android.widget.Toast
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException

object ImageUploader {
    fun uploadImageFromAssets(context: Context, assetFileName: String) {
        val assetManager = context.assets
        val inputStream = assetManager.open(assetFileName)
        val fileBytes = inputStream.readBytes()

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("img1", assetFileName, RequestBody.create("image/*".toMediaTypeOrNull(), fileBytes))
            .build()

        val request = Request.Builder()
            .url("http://10.0.2.2:5000/verify-image")
            .post(requestBody)
            .build()

        // Customizing OkHttpClient with a longer read timeout
        val client = OkHttpClient.Builder()
            .readTimeout(1000, java.util.concurrent.TimeUnit.SECONDS) // Set the read timeout to 60 seconds
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                (context as Activity).runOnUiThread {
                    Toast.makeText(context, "Network error or cannot reach the server", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                (context as Activity).runOnUiThread {
                    when (response.code) {
                        200 -> Toast.makeText(context, "Verified", Toast.LENGTH_SHORT).show()
                        400 -> Toast.makeText(context, "Unverified", Toast.LENGTH_SHORT).show()
                        500 -> Toast.makeText(context, "Server error", Toast.LENGTH_SHORT).show()
                        else -> Toast.makeText(context, "Unexpected response code: ${response.code}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }
}
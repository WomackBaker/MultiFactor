package com.example.multifactorapp

import android.app.Activity
import android.content.Context
import android.net.Uri
import android.widget.Toast
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.File
import java.io.IOException

object ImageUploader {
    fun uploadImage(context: Context, imageUri: Uri) {
        val file = File(imageUri.path) // Make sure you have the correct path here.
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", file.name, RequestBody.create("image/*".toMediaTypeOrNull(), file))
            .build()

        val request = Request.Builder()
            .url("http://127.0.0.1:5000/verify-image")
            .post(requestBody)
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // This callback is for network failures or request preparation failures.
                (context as Activity).runOnUiThread {
                    Toast.makeText(context, "Network error or cannot reach the server", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                // This callback means the server responded, regardless of the HTTP status code.
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

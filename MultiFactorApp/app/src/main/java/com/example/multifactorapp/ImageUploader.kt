package com.example.multifactorapp

import android.app.Activity
import android.content.Context
import android.widget.Toast
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException

/**
 * Provides functionality to upload images to a server.
 * This object handles the uploading of image files from the application's assets.
 */
object ImageUploader {

    /**
     * Uploads an image file from the application's assets to a remote server.
     *
     * @param context The context of the calling activity, used for accessing assets and displaying Toasts.
     * @param assetFileName The name of the file in the assets directory to be uploaded.
     */
    fun uploadImageFromAssets(context: Context, assetFileName: String) {
        // Access the application's asset manager to open the specified asset file
        val assetManager = context.assets
        val inputStream = assetManager.open(assetFileName)
        val fileBytes = inputStream.readBytes() // Read the entire content of the file into a byte array

        // Construct a multipart request body with the image data
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("img1", assetFileName, RequestBody.create("image/*".toMediaTypeOrNull(), fileBytes))
            .build()

        // Build the HTTP request to upload the image
        val request = Request.Builder()
            .url("http://10.0.2.2:30080/verify-image")
            .post(requestBody)
            .build()

        // Customize OkHttpClient with a longer read timeout to accommodate potentially large uploads
        val client = OkHttpClient.Builder()
            .readTimeout(1000, java.util.concurrent.TimeUnit.SECONDS) // Set the read timeout to 1000 seconds
            .build()

        // Asynchronously send the HTTP request
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // Handle failure in network communication or server response
                (context as Activity).runOnUiThread {
                    Toast.makeText(context, "Network error or cannot reach the server", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                // Handle the server's response on the UI thread
                (context as Activity).runOnUiThread {
                    when (response.code) {
                        200 -> Toast.makeText(context, "Image successfully verified", Toast.LENGTH_SHORT).show()
                        400 -> Toast.makeText(context, "Verification failed", Toast.LENGTH_SHORT).show()
                        500 -> Toast.makeText(context, "Server encountered an error", Toast.LENGTH_SHORT).show()
                        else -> Toast.makeText(context, "Unexpected response code: ${response.code}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }
}

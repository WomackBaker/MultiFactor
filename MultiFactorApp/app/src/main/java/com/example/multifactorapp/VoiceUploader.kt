import android.app.Activity
import android.content.Context
import android.widget.Toast
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException

/**
 * Handles the uploading of voice files to a server.
 * This object is designed to upload voice files stored in the application's assets directory.
 */
object VoiceUploader {

    /**
     * Uploads a voice file from the application's assets to a remote server.
     * This method prepares and sends an HTTP POST request containing the voice file.
     *
     * @param context The context of the calling activity, used for accessing assets and displaying Toasts.
     * @param assetFileName The name of the file in the assets directory to be uploaded.
     */
    fun uploadVoiceFromAssets(context: Context, assetFileName: String) {
        try {
            // Access the application's asset manager to open and read the specified voice file
            val assetManager = context.assets
            val inputStream = assetManager.open(assetFileName)
            val fileBytes = inputStream.readBytes()
            inputStream.close()  // Ensure the input stream is closed after use

            // Construct a multipart request body with the voice data and additional form data
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("voice", assetFileName,
                    RequestBody.create("audio/*".toMediaTypeOrNull(), fileBytes))
                .addFormDataPart("name", "Baker")  // Additional form data can be included here
                .build()

            // Build the HTTP request to upload the voice file
            val request = Request.Builder()
                .url("http://10.0.2.2:30080/verify-voice")
                .post(requestBody)
                .build()

            // Create and enqueue the HTTP request asynchronously
            OkHttpClient().newCall(request).enqueue(object : Callback {
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
                            200 -> Toast.makeText(context, "Voice successfully verified", Toast.LENGTH_SHORT).show()
                            400 -> Toast.makeText(context, "Verification failed", Toast.LENGTH_SHORT).show()
                            500 -> Toast.makeText(context, "Server encountered an error", Toast.LENGTH_SHORT).show()
                            else -> Toast.makeText(context, "Unexpected response code: ${response.code}", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            })
        } catch (e: IOException) {
            // Handle errors in file access
            e.printStackTrace()
            (context as Activity).runOnUiThread {
                Toast.makeText(context, "Error accessing asset file", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

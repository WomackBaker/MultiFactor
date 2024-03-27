import android.app.Activity
import android.content.Context
import android.widget.Toast
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException

object VoiceUploader {
    fun uploadVoiceFromAssets(context: Context, assetFileName: String) {
        try {
            val assetManager = context.assets
            val inputStream = assetManager.open(assetFileName)
            val fileBytes = inputStream.readBytes()
            inputStream.close()

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("voice", assetFileName,
                    RequestBody.create("audio/*".toMediaTypeOrNull(), fileBytes))
                .addFormDataPart("name", "Baker")
                .build()

            val request = Request.Builder()
                .url("http://10.0.2.2:5000/verify-voice")
                .post(requestBody)
                .build()

            OkHttpClient().newCall(request).enqueue(object : Callback {
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
        } catch (e: IOException) {
            e.printStackTrace()
            (context as Activity).runOnUiThread {
                Toast.makeText(context, "Error accessing asset file", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

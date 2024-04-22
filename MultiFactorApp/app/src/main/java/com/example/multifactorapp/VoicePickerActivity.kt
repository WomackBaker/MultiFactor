package com.example.multifactorapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * An activity dedicated to handling the selection and uploading of a voice file.
 * It automatically uploads a predetermined voice file from the assets and closes.
 */
class VoicePickerActivity : AppCompatActivity() {

    /**
     * Called when the activity is starting. This is where most initialization should go:
     * calling setContentView(int) to inflate the activity's UI, using findViewById(int)
     * to programmatically interact with widgets in the UI, etc.
     *
     * @param savedInstanceState If the activity is being re-initialized after previously being
     * shut down then this Bundle contains the data it most recently supplied in onSaveInstanceState(Bundle).
     * Note: Otherwise it is null.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        uploadVoiceAndFinish() // Calls the method to upload a voice file and then finishes the activity.
    }

    /**
     * Uploads a voice file from the assets directory and then closes the activity.
     * Specifically, it uploads a file named 'voice.wav' located within a folder named 'audio'
     * in the assets directory. After uploading the voice file, it terminates the activity.
     */
    private fun uploadVoiceAndFinish() {
        VoiceUploader.uploadVoiceFromAssets(this, "audio/voice.wav") // Uploads the voice file.
        finish() // Closes the activity after the upload is initiated.
    }

    companion object {
        /**
         * Starts this activity from another activity context.
         * This static method provides a convenient way to create an intent, start this activity,
         * and potentially clear it from the recent activity list after use.
         *
         * @param activity The context of the calling activity.
         */
        fun start(activity: Activity) {
            val intent = Intent(activity, VoicePickerActivity::class.java)
            activity.startActivity(intent) // Initiates the transition to this activity.
        }
    }
}

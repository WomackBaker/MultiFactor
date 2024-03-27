package com.example.multifactorapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class VoicePickerActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        uploadVoiceAndFinish()
    }

    private fun uploadVoiceAndFinish() {
        VoiceUploader.uploadVoiceFromAssets(this, "audio/voice.wav")
        finish()
    }

    companion object {
        fun start(activity: Activity) {
            val intent = Intent(activity, VoicePickerActivity::class.java)
            activity.startActivity(intent)
        }
    }
}

package com.example.multifactorapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class ImagePickerActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        uploadImageAndFinish()
    }

    private fun uploadImageAndFinish() {
        // Directly upload 'randomperson.jpg' from assets
        ImageUploader.uploadImageFromAssets(this, "images/randomperson.jpg")
        finish()
    }

    companion object {
        fun start(activity: Activity) {
            val intent = Intent(activity, ImagePickerActivity::class.java)
            activity.startActivity(intent)
        }
    }
}
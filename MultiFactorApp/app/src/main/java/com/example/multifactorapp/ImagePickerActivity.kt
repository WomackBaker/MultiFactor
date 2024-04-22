package com.example.multifactorapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * An activity that handles the selection and uploading of an image.
 */
class ImagePickerActivity : AppCompatActivity() {

    /**
     * Called when the activity is starting.
     * This is where most initialization should go: calling setContentView(int) to inflate
     * the activity's UI, using findViewById(int) to programmatically interact with widgets
     * in the UI, calling managedQuery(android.net.Uri, String[], String, String[], String)
     * to retrieve cursors for data being displayed, etc.
     *
     * @param savedInstanceState If the activity is being re-initialized after previously being
     * shut down then this Bundle contains the data it most recently supplied in onSaveInstanceState(Bundle).
     * Note: Otherwise it is null.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        uploadImageAndFinish()
    }

    /**
     * Uploads an image from the assets directory and finishes the activity.
     * The method specifically uploads a file named 'randomperson.jpg' located within a folder named 'images'
     * in the assets directory. After uploading the image, it closes the activity.
     */
    private fun uploadImageAndFinish() {
        // Directly upload 'randomperson.jpg' from assets
        ImageUploader.uploadImageFromAssets(this, "images/randomperson.jpg")
        // Terminate the activity
        finish()
    }

    companion object {
        /**
         * Starts this activity from another activity context.
         * This static method provides a convenient way to create an intent, start this activity,
         * and clear it from the recent activity list.
         *
         * @param activity The context of the calling activity.
         */
        fun start(activity: Activity) {
            val intent = Intent(activity, ImagePickerActivity::class.java)
            activity.startActivity(intent)
        }
    }
}

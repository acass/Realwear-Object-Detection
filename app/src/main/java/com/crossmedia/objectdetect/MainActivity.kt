package com.crossmedia.objectdetect

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.FrameLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ObjectDetect"
        private const val PERMISSION_REQUEST = 10
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var pauseButton: Button
    private var detector: Detector? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var paused = false
    private var overlaySized = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        pauseButton = findViewById(R.id.pauseButton)

        pauseButton.setOnClickListener {
            paused = !paused
            pauseButton.setText(if (paused) R.string.resume_detection else R.string.pause_detection)
            if (paused) overlayView.setDetections(emptyList())
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), PERMISSION_REQUEST)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun startCamera() {
        analysisExecutor.execute {
            val d = Detector(this)
            runOnUiThread { detector = d; bindCamera() }
        }
    }

    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { image -> analyze(image) }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        image.use {
            if (paused) return
            val d = detector ?: return

            var bitmap = it.toBitmap()
            val rotation = it.imageInfo.rotationDegrees
            if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
            }

            // Center-crop to square for the square model input
            val side = min(bitmap.width, bitmap.height)
            val square = Bitmap.createBitmap(
                bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side
            )

            if (!overlaySized) {
                overlaySized = true
                runOnUiThread { sizeOverlay(bitmap.width, bitmap.height) }
            }

            val start = System.nanoTime()
            val detections = d.detect(square)
            val ms = (System.nanoTime() - start) / 1_000_000
            Log.d(TAG, "Inference ${ms}ms, ${detections.size} detections")

            runOnUiThread { if (!paused) overlayView.setDetections(detections) }
        }
    }

    /**
     * The preview uses FILL_CENTER (cover). The model sees the center square of
     * the analysis frame, which shares its center with the preview crop, so the
     * overlay is a centered square scaled by the cover factor.
     */
    private fun sizeOverlay(imgW: Int, imgH: Int) {
        val vw = previewView.width.toFloat()
        val vh = previewView.height.toFloat()
        if (vw == 0f || vh == 0f) { overlaySized = false; return }
        val scale = max(vw / imgW, vh / imgH)
        val sideOnScreen = (min(imgW, imgH) * scale).toInt()
        overlayView.layoutParams = FrameLayout.LayoutParams(
            sideOnScreen, sideOnScreen, android.view.Gravity.CENTER
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        detector?.close()
    }
}

package com.crossmedia.objectdetect

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
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

        /**
         * Builds the rotate + centre-crop + scale transform from a [srcW] x [srcH]
         * camera frame onto a [size] x [size] model input, writing it [into].
         *
         * Scaling about the centre by `size / min(rotatedW, rotatedH)` and drawing
         * onto a square canvas is exactly a centre-crop followed by a scale: the
         * shorter axis lands flush on the canvas edges, the longer one overflows
         * and is clipped. Kept separate from the draw so it can be checked without
         * a camera.
         */
        internal fun cropMatrix(srcW: Int, srcH: Int, rotation: Int, size: Int, into: Matrix) {
            // Rotation permutes the two axes, and min is symmetric, so the shorter
            // side is the same before and after rotating - no swap needed here.
            val scale = size.toFloat() / min(srcW, srcH)

            into.reset()
            into.postTranslate(-srcW / 2f, -srcH / 2f)
            into.postRotate(rotation.toFloat())
            into.postScale(scale, scale)
            into.postTranslate(size / 2f, size / 2f)
        }
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var pauseButton: Button

    // Confined to analysisExecutor: created there, used there, closed there. The
    // single thread orders close() after any in-flight detect(), so no frame can
    // be running against a freed interpreter.
    private var detector: Detector? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var paused = false
    @Volatile private var overlaySized = false

    // Analysis-thread scratch, allocated once. Reused so a steady-state frame
    // allocates nothing: see analyze().
    private var sourceBitmap: Bitmap? = null
    private var modelBitmap: Bitmap? = null
    private var modelCanvas: Canvas? = null
    private val frameMatrix = Matrix()
    private val framePaint = Paint(Paint.FILTER_BITMAP_FLAG)

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
            // Assigned on the analysis thread that reads it, so the field is never
            // shared across threads. Only the camera binding needs the UI thread.
            detector = Detector(this)
            runOnUiThread { bindCamera() }
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

            val square = prepareFrame(it, d.inputSize)

            if (!overlaySized) {
                overlaySized = true
                // Overlay geometry is in rotated frame space, which is what the
                // preview shows.
                val rotated = it.imageInfo.rotationDegrees % 180 != 0
                val w = if (rotated) it.height else it.width
                val h = if (rotated) it.width else it.height
                runOnUiThread { sizeOverlay(w, h) }
            }

            val start = System.nanoTime()
            val detections = d.detect(square)
            val ms = (System.nanoTime() - start) / 1_000_000
            Log.d(TAG, "Inference ${ms}ms, ${detections.size} detections")

            runOnUiThread { if (!paused) overlayView.setDetections(detections) }
        }
    }

    /**
     * Rotates, centre-crops to square and scales the frame to [size] in a single
     * draw into a reusable bitmap.
     *
     * Doing it as one matrix rather than three chained `Bitmap.createBitmap`
     * calls is what keeps a steady-state frame allocation-free. Scaling about the
     * centre by `size / min(rotatedW, rotatedH)` and drawing onto a size x size
     * canvas is exactly a centre-crop followed by a scale: the shorter axis lands
     * flush on the canvas and the longer axis overflows and is clipped.
     */
    private fun prepareFrame(image: ImageProxy, size: Int): Bitmap {
        val src = sourceBitmap(image)

        val out = modelBitmap ?: Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
            .also { modelBitmap = it; modelCanvas = Canvas(it) }

        cropMatrix(src.width, src.height, image.imageInfo.rotationDegrees, size, frameMatrix)
        modelCanvas!!.drawBitmap(src, frameMatrix, framePaint)
        return out
    }

    /**
     * The camera frame as a bitmap, reusing one allocation where the plane is
     * tightly packed. Padded rows are rare but legal, so fall back to CameraX's
     * own conversion rather than decoding the stride by hand.
     */
    private fun sourceBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        if (plane.rowStride != image.width * 4) return image.toBitmap()

        val reuse = sourceBitmap?.takeIf { it.width == image.width && it.height == image.height }
            ?: Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                .also { sourceBitmap = it }
        plane.buffer.rewind()
        reuse.copyPixelsFromBuffer(plane.buffer)
        return reuse
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
        // Queue the close behind any frame already running, then stop accepting
        // work. shutdown() alone does not wait, so closing here would be a
        // use-after-free on the interpreter's native memory.
        analysisExecutor.execute {
            detector?.close()
            detector = null
        }
        analysisExecutor.shutdown()
    }
}

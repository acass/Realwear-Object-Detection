package com.crossmedia.objectdetect

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
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
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ObjectDetect"
        private const val PERMISSION_REQUEST = 10
        private const val UTTERANCE_ID = "procedure"

        // High contrast on the summary's black backing, which matters more on a
        // monocular headset display than matching any palette.
        private val FOUND_COLOR = 0xFF00E676.toInt()
        private val MISSING_COLOR = 0xFFFF5252.toInt()

        // Spoken step numbers. Only ever indexed 0..results.size, which the
        // procedure caps at five.
        private val NUMBER_WORDS = arrayOf("zero", "one", "two", "three", "four", "five")

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
    private lateinit var startButton: Button
    private lateinit var goodButton: Button
    private lateinit var notFoundButton: Button
    private lateinit var backButton: Button
    private lateinit var restartButton: Button
    private lateinit var exitButton: Button
    private lateinit var stepBanner: TextView
    private lateinit var summaryText: TextView

    // Confined to analysisExecutor: created there, used there, closed there. The
    // single thread orders close() after any in-flight detect(), so no frame can
    // be running against a freed interpreter.
    private var detector: Detector? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var paused = false
    @Volatile private var overlaySized = false

    // UI-thread only. The analysis thread never touches it; it reads targetLabel
    // and posts its measurement back, which applyMode() keeps in step.
    private val procedure = ProcedureState(ProcedureState.DEFAULT_TARGETS)

    // The one field shared with the analysis thread, same pattern as paused:
    // written on the UI thread whenever the step changes, read per frame.
    // Null means free-running detection - draw everything, record nothing.
    @Volatile private var targetLabel: String? = null

    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var pendingUtterance: String? = null

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
        startButton = findViewById(R.id.startButton)
        goodButton = findViewById(R.id.goodButton)
        notFoundButton = findViewById(R.id.notFoundButton)
        backButton = findViewById(R.id.backButton)
        restartButton = findViewById(R.id.restartButton)
        exitButton = findViewById(R.id.exitButton)
        stepBanner = findViewById(R.id.stepBanner)
        summaryText = findViewById(R.id.summaryText)

        pauseButton.setOnClickListener {
            paused = !paused
            pauseButton.setText(if (paused) R.string.resume_detection else R.string.pause_detection)
            if (paused) overlayView.setDetections(emptyList())
        }

        // Every one of these is a state transition, so applyMode() is the whole
        // handler: it redraws the screen from whatever the procedure now says.
        startButton.setOnClickListener { procedure.start(); applyMode() }
        goodButton.setOnClickListener { procedure.answer(true); applyMode() }
        notFoundButton.setOnClickListener { procedure.answer(false); applyMode() }
        backButton.setOnClickListener { procedure.goBack(); applyMode() }
        restartButton.setOnClickListener { procedure.start(); applyMode() }
        exitButton.setOnClickListener { procedure.exit(); applyMode() }

        initTts()
        applyMode()

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

            // Inside a step only the target is drawn, and its strongest hit this
            // frame goes back into the step's record.
            val target = targetLabel
            val shown = if (target == null) detections else detections.filter { det -> det.label == target }
            val peak = if (target == null) 0f else shown.maxOfOrNull { det -> det.confidence } ?: 0f

            runOnUiThread {
                if (paused) return@runOnUiThread
                // The step can advance between reading targetLabel and this post
                // landing. Re-check on the UI thread so a stale frame is never
                // credited to the step that follows it - targets are distinct, so
                // a matching label means the same step is still open.
                if (target != null && procedure.currentTarget == target) procedure.observe(peak)
                overlayView.setDetections(shown)
            }
        }
    }

    /**
     * The one place mode-dependent state is written: every button's visibility,
     * the banner, the summary, the analysis thread's target and the paused flag
     * all come from [ProcedureState.mode] here, so none of them can drift apart.
     *
     * Called only on transitions, which is what makes speaking from here right -
     * one utterance per step change, none on a redraw.
     */
    private fun applyMode() {
        val idle = procedure.mode == Mode.IDLE
        val running = procedure.mode == Mode.RUNNING
        val summary = procedure.mode == Mode.SUMMARY

        // Only visible buttons become WearHF voice commands, so this is also the
        // definition of what can be said in each mode.
        pauseButton.visibility = if (idle) View.VISIBLE else View.GONE
        startButton.visibility = if (idle) View.VISIBLE else View.GONE
        goodButton.visibility = if (running) View.VISIBLE else View.GONE
        notFoundButton.visibility = if (running) View.VISIBLE else View.GONE
        backButton.visibility = if (running) View.VISIBLE else View.GONE
        restartButton.visibility = if (summary) View.VISIBLE else View.GONE
        exitButton.visibility = if (summary) View.VISIBLE else View.GONE

        stepBanner.visibility = if (running) View.VISIBLE else View.GONE
        summaryText.visibility = if (summary) View.VISIBLE else View.GONE

        targetLabel = procedure.currentTarget
        // The summary is a frozen screen, so stop inference through the existing
        // pause path rather than adding a second way to halt the analyser.
        paused = summary
        overlayView.setDetections(emptyList())

        when (procedure.mode) {
            Mode.IDLE -> pauseButton.setText(R.string.pause_detection)

            Mode.RUNNING -> {
                val step = procedure.stepIndex + 1
                val target = procedure.currentTarget!!
                stepBanner.text =
                    "STEP $step OF ${procedure.results.size} — FIND THE ${target.uppercase(Locale.US)}"
                speak("Step ${NUMBER_WORDS[step]}. Look for the $target.")
            }

            Mode.SUMMARY -> {
                summaryText.text = summaryLines()
                val found = procedure.foundCount
                speak(
                    "Procedure complete. ${NUMBER_WORDS[found]} of " +
                        "${NUMBER_WORDS[procedure.results.size]} found."
                )
            }
        }
    }

    /**
     * The result list: the operator's verdict in green or red, and beside it what
     * the detector actually managed to see. Five fixed rows, so a span-coloured
     * TextView beats a list adapter.
     */
    private fun summaryLines(): CharSequence {
        val sb = SpannableStringBuilder()
        for ((i, r) in procedure.results.withIndex()) {
            if (i > 0) sb.append('\n')
            val found = r.foundByOperator == true

            val verdictStart = sb.length
            sb.append(r.label.uppercase(Locale.US))
            sb.append(if (found) "  FOUND" else "  NOT FOUND")
            sb.setSpan(
                ForegroundColorSpan(if (found) FOUND_COLOR else MISSING_COLOR),
                verdictStart, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )

            // peakConfidence is 0 only when the target never cleared the
            // detector's threshold, so it doubles as "never seen".
            sb.append(
                if (r.peakConfidence > 0f)
                    String.format(Locale.US, "   (model %.2f)", r.peakConfidence)
                else "   (model never)"
            )
        }
        return sb
    }

    private fun initTts() {
        tts = TextToSpeech(this) { status ->
            if (status != TextToSpeech.SUCCESS) {
                Log.w(TAG, "TextToSpeech unavailable ($status), prompts are on-screen only")
                return@TextToSpeech
            }
            // The callback can arrive before the constructor has returned and
            // assigned tts, so touch the engine from a posted runnable rather
            // than here. post() always queues, unlike runOnUiThread.
            overlayView.post {
                tts?.language = Locale.US
                ttsReady = true
                pendingUtterance?.let { speak(it) }
                pendingUtterance = null
            }
        }
    }

    /**
     * Speaks a prompt, holding it until the engine finishes initialising - a cold
     * start would otherwise swallow the very first step's prompt.
     *
     * QUEUE_FLUSH so a fast "Good" cuts the previous prompt off instead of making
     * the operator wait through it.
     */
    private fun speak(text: String) {
        val engine = tts
        if (!ttsReady || engine == null) {
            pendingUtterance = text
            return
        }
        engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
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
        tts?.stop()
        tts?.shutdown()
        tts = null
        ttsReady = false
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

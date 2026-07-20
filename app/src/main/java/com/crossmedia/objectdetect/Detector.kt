package com.crossmedia.objectdetect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class Detection(val box: RectF, val label: String, val confidence: Float)

/**
 * Runs YOLOv8 TFLite inference. Handles both float32 and int8-quantized
 * input/output tensors, since Ultralytics exports vary by flags.
 * Box coordinates in [Detection.box] are normalized 0..1 relative to the
 * square model input.
 */
class Detector(context: Context) {

    companion object {
        private const val TAG = "Detector"
        private const val MODEL_FILE = "yolov8n_int8.tflite"
        private const val LABELS_FILE = "labels.txt"
        const val CONFIDENCE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.45f
    }

    private val labels: List<String> =
        context.assets.open(LABELS_FILE).bufferedReader().readLines().filter { it.isNotBlank() }

    private var nnApiDelegate: NnApiDelegate? = null
    private val modelBuffer: java.nio.MappedByteBuffer =
        context.assets.openFd(MODEL_FILE).use { fd ->
            fd.createInputStream().channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    private var interpreter: Interpreter = createInterpreter()
    private var checkedSpeed = false

    val inputSize: Int
    private val inputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val numChannels: Int   // 84 for COCO (4 box + 80 classes)
    private val numBoxes: Int
    private val outputType: DataType
    private val outputScale: Float
    private val outputZeroPoint: Int

    private val inputBuffer: ByteBuffer
    private val pixels: IntArray

    init {
        val inTensor = interpreter.getInputTensor(0)
        inputSize = inTensor.shape()[1]          // [1, H, W, 3]
        inputType = inTensor.dataType()
        inputScale = inTensor.quantizationParams().scale
        inputZeroPoint = inTensor.quantizationParams().zeroPoint

        val outTensor = interpreter.getOutputTensor(0)  // [1, 84, N]
        numChannels = outTensor.shape()[1]
        numBoxes = outTensor.shape()[2]
        outputType = outTensor.dataType()
        outputScale = outTensor.quantizationParams().scale
        outputZeroPoint = outTensor.quantizationParams().zeroPoint

        val bytesPerElem = if (inputType == DataType.FLOAT32) 4 else 1
        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * bytesPerElem)
            .order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)

        Log.i(TAG, "Model loaded: input ${inputSize}x$inputSize $inputType, output ${numChannels}x$numBoxes $outputType")
    }

    // ponytail: NNAPI then CPU; add GPU delegate only if measured FPS is short
    private fun createInterpreter(): Interpreter {
        try {
            val delegate = NnApiDelegate()
            val interp = Interpreter(modelBuffer, Interpreter.Options().addDelegate(delegate))
            nnApiDelegate = delegate
            Log.i(TAG, "Using NNAPI delegate")
            return interp
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI unavailable, falling back to CPU", e)
        }
        return createCpuInterpreter()
    }

    private fun createCpuInterpreter(): Interpreter =
        Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))

    /** [bitmap] must already be square; it is scaled to the model input size. */
    fun detect(bitmap: Bitmap): List<Detection> {
        val scaled = if (bitmap.width == inputSize && bitmap.height == inputSize) bitmap
        else Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.rewind()
        if (inputType == DataType.FLOAT32) {
            for (p in pixels) {
                inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((p and 0xFF) / 255f)
            }
        } else {
            for (p in pixels) {
                inputBuffer.put(quantize(((p shr 16) and 0xFF) / 255f))
                inputBuffer.put(quantize(((p shr 8) and 0xFF) / 255f))
                inputBuffer.put(quantize((p and 0xFF) / 255f))
            }
        }

        if (!checkedSpeed && nnApiDelegate != null) {
            checkedSpeed = true
            val start = System.nanoTime()
            runInference()
            val ms = (System.nanoTime() - start) / 1_000_000
            if (ms > 1500) {
                // NNAPI driver is a slow software fallback (e.g. emulator); use CPU instead
                Log.w(TAG, "NNAPI inference took ${ms}ms, switching to CPU")
                interpreter.close()
                nnApiDelegate?.close()
                nnApiDelegate = null
                interpreter = createCpuInterpreter()
            }
        }
        val raw: Array<FloatArray> = runInference()
        return postProcess(raw)
    }

    private fun quantize(v: Float): Byte =
        (v / inputScale + inputZeroPoint).toInt().coerceIn(-128, 127).toByte()

    /** Returns output as [numChannels][numBoxes] floats, dequantized if needed. */
    private fun runInference(): Array<FloatArray> {
        inputBuffer.rewind()
        val out = Array(numChannels) { FloatArray(numBoxes) }
        if (outputType == DataType.FLOAT32) {
            val outBuf = arrayOf(out)
            interpreter.run(inputBuffer, outBuf)
            return out
        }
        val byteOut = ByteBuffer.allocateDirect(numChannels * numBoxes).order(ByteOrder.nativeOrder())
        interpreter.run(inputBuffer, byteOut)
        byteOut.rewind()
        for (c in 0 until numChannels) {
            for (b in 0 until numBoxes) {
                out[c][b] = (byteOut.get().toInt() - outputZeroPoint) * outputScale
            }
        }
        return out
    }

    private fun postProcess(out: Array<FloatArray>): List<Detection> {
        val candidates = ArrayList<Detection>()
        // Ultralytics TFLite exports normalize coords to 0..1; guard for pixel-space models.
        var coordMax = 0f
        for (b in 0 until numBoxes) {
            if (out[2][b] > coordMax) coordMax = out[2][b]
        }
        val coordDiv = if (coordMax > 1.5f) inputSize.toFloat() else 1f

        for (b in 0 until numBoxes) {
            var bestClass = -1
            var bestScore = 0f
            for (c in 4 until numChannels) {
                val s = out[c][b]
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c - 4
                }
            }
            if (bestScore < CONFIDENCE_THRESHOLD) continue

            val cx = out[0][b] / coordDiv
            val cy = out[1][b] / coordDiv
            val w = out[2][b] / coordDiv
            val h = out[3][b] / coordDiv
            candidates.add(
                Detection(
                    RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    labels.getOrElse(bestClass) { "class $bestClass" },
                    bestScore
                )
            )
        }
        return nms(candidates)
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = ArrayList<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.box, it.box) > IOU_THRESHOLD }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        interpreter.close()
        nnApiDelegate?.close()
    }
}

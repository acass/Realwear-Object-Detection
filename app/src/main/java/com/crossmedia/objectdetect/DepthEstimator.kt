package com.crossmedia.objectdetect

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Monocular depth estimation with yolo26n-depth exported to LiteRT.
 *
 * Output is metres. The head computes exp(clamp(logit, -4, 5)) then applies a baked
 * log-affine calibration (cal_a = 1.0, cal_b = -0.19385, i.e. a x0.8238 global scale)
 * that Ultralytics fit at imgsz=768 on their own data. We run at 320 against a camera
 * whose FOV is not published, so [SCALE_CORRECTION] exists to absorb the difference and
 * must be tuned against a tape measure on the device.
 *
 * Unlike the detector, this graph is NCHW and planar: the input buffer holds every red
 * value, then every green, then every blue -- not interleaved per pixel.
 */
class DepthEstimator(context: Context) {

    companion object {
        private const val TAG = "DepthEstimator"
        private const val MODEL_FILE = "yolo26n-depth.tflite"
        private const val BENCHMARK_RUNS = 3

        /**
         * Global multiplier applied to raw model output.
         *
         * UNCALIBRATED. Tune against a tape measure on the target device and record the
         * date and device here; see the calibration step in the plan. Until then the
         * displayed metres carry the model's own 768px-fit scale, which is plausible but
         * unverified for this camera.
         */
        const val SCALE_CORRECTION = 1.0f
    }

    /** Whichever delegate chooseDelegate() kept, or null if CPU won. */
    private var delegateHandle: AutoCloseable? = null

    private val modelBuffer: java.nio.MappedByteBuffer =
        context.assets.openFd(MODEL_FILE).use { fd ->
            fd.createInputStream().channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }

    // Starts on CPU so tensor shapes can be read without committing to a delegate.
    private var interpreter: Interpreter = createCpuInterpreter()
    private var benchmarked = false

    val inputSize: Int
    private val channelsFirst: Boolean   // true for [1,3,H,W], false for [1,H,W,3]
    val outputWidth: Int
    val outputHeight: Int

    private val inputBuffer: ByteBuffer
    private val pixels: IntArray

    /** Reused every frame; [Interpreter.run] overwrites it wholesale. */
    private val outputBuffer: FloatBuffer
    val depth: FloatArray

    init {
        val inShape = interpreter.getInputTensor(0).shape()
        channelsFirst = inShape[1] == 3
        inputSize = if (channelsFirst) inShape[2] else inShape[1]

        // [1,1,H,W] or [1,H,W,1] -- single channel, so take the two largest dims.
        val outShape = interpreter.getOutputTensor(0).shape()
        val spatial = outShape.filter { it > 1 }
        outputHeight = spatial.getOrElse(0) { inputSize }
        outputWidth = spatial.getOrElse(1) { inputSize }

        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)

        depth = FloatArray(outputWidth * outputHeight)
        outputBuffer = ByteBuffer.allocateDirect(depth.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()

        Log.i(TAG, "Depth model loaded: input ${inShape.contentToString()} " +
            "(${if (channelsFirst) "NCHW" else "NHWC"}), output ${outShape.contentToString()}")
    }

    private fun createCpuInterpreter(): Interpreter =
        Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))

    private fun chooseDelegate() {
        val choice = DelegateRace.run(TAG, modelBuffer, interpreter) { timeInference(it) }
        interpreter = choice.interpreter
        delegateHandle = choice.delegate
    }

    /** Fastest of [BENCHMARK_RUNS] inferences on the frame already in [inputBuffer]. */
    private fun timeInference(target: Interpreter): Long {
        var best = Long.MAX_VALUE
        repeat(BENCHMARK_RUNS) {
            val start = System.nanoTime()
            runInference(target)
            best = minOf(best, (System.nanoTime() - start) / 1_000_000)
        }
        return best
    }

    /**
     * Returns the depth map in metres, row-major, [outputWidth] x [outputHeight].
     * The returned array is [depth] and is overwritten on the next call.
     * [bitmap] must already be square.
     */
    fun estimate(bitmap: Bitmap): FloatArray {
        val scaled = if (bitmap.width == inputSize && bitmap.height == inputSize) bitmap
        else Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.rewind()
        if (channelsFirst) {
            // Planar: every R, then every G, then every B.
            for (shift in intArrayOf(16, 8, 0)) {
                for (p in pixels) inputBuffer.putFloat(((p shr shift) and 0xFF) / 255f)
            }
        } else {
            for (p in pixels) {
                inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
                inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
                inputBuffer.putFloat((p and 0xFF) / 255f)
            }
        }

        if (!benchmarked) {
            benchmarked = true
            chooseDelegate()
        }
        return runInference()
    }

    private fun runInference(target: Interpreter = interpreter): FloatArray {
        inputBuffer.rewind()
        outputBuffer.rewind()
        target.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        outputBuffer.get(depth)
        if (SCALE_CORRECTION != 1.0f) {
            for (i in depth.indices) depth[i] *= SCALE_CORRECTION
        }
        return depth
    }

    fun close() {
        interpreter.close()
        delegateHandle?.close()
    }
}

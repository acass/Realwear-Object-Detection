package com.crossmedia.objectdetect

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One detected person: [YoloPostProcessor.KEYPOINT_COUNT] triples of
 * (x, y, confidence), normalized 0..1 over the square model input.
 *
 * Not a data class: the generated equals would compare the keypoint array by
 * identity, which is a quiet trap for anything doing contains() or distinct().
 */
class Pose(val keypoints: FloatArray, val score: Float)

/**
 * Runs YOLO pose TFLite inference. Handles both float32 and int8-quantized
 * input/output tensors, since Ultralytics exports vary by flags.
 */
class Detector(context: Context) {

    companion object {
        private const val TAG = "Detector"
        private const val MODEL_FILE = "yolo26n_pose_fp32.tflite"
        const val CONFIDENCE_THRESHOLD = 0.5f
        const val IOU_THRESHOLD = 0.45f
        /** Gates drawing, not decoding: an occluded joint should vanish, not snap to a corner. */
        const val KEYPOINT_THRESHOLD = 0.5f
        private const val BENCHMARK_RUNS = 3
        /** ARGB right-shifts in model channel order: R, G, B. */
        private val RGB_SHIFTS = intArrayOf(16, 8, 0)
    }

    private var nnApiDelegate: NnApiDelegate? = null
    private val modelBuffer: java.nio.MappedByteBuffer =
        context.assets.openFd(MODEL_FILE).use { fd ->
            fd.createInputStream().channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    // Starts on CPU so tensor shapes can be read without committing to a delegate;
    // the first detect() benchmarks both and keeps the winner. See chooseDelegate().
    private var interpreter: Interpreter = createCpuInterpreter()
    private var benchmarked = false

    val inputSize: Int
    private val channelsFirst: Boolean
    private val inputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val numChannels: Int   // 56 for pose (4 box + 1 person score + 17 keypoint triples)
    private val numBoxes: Int
    private val outputType: DataType
    private val outputScale: Float
    private val outputZeroPoint: Int

    private val inputBuffer: ByteBuffer
    private val pixels: IntArray
    private val postProcessor: YoloPostProcessor

    // Reused every frame. interpreter.run overwrites outputArray wholesale, and
    // quantizedOutput is rewound before each run, so neither needs reallocating.
    private val outputArray: Array<FloatArray>
    private val outputHolder: Array<Array<FloatArray>>
    private val quantizedOutput: ByteBuffer?

    init {
        val inTensor = interpreter.getInputTensor(0)
        // Ultralytics' LiteRT exporter keeps PyTorch's [1, 3, H, W]; the older
        // TensorFlow converter emitted [1, H, W, 3]. Which one you get decides
        // whether the frame is written planar or interleaved, and getting it
        // wrong feeds the model scrambled pixels rather than failing loudly.
        val inShape = inTensor.shape()
        channelsFirst = inShape[1] == 3
        inputSize = if (channelsFirst) inShape[2] else inShape[1]
        inputType = inTensor.dataType()
        inputScale = inTensor.quantizationParams().scale
        inputZeroPoint = inTensor.quantizationParams().zeroPoint

        val outTensor = interpreter.getOutputTensor(0)  // [1, 56, N]
        numChannels = outTensor.shape()[1]
        numBoxes = outTensor.shape()[2]
        outputType = outTensor.dataType()
        outputScale = outTensor.quantizationParams().scale
        outputZeroPoint = outTensor.quantizationParams().zeroPoint

        val bytesPerElem = if (inputType == DataType.FLOAT32) 4 else 1
        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * bytesPerElem)
            .order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)

        outputArray = Array(numChannels) { FloatArray(numBoxes) }
        outputHolder = arrayOf(outputArray)
        quantizedOutput =
            if (outputType == DataType.FLOAT32) null
            else ByteBuffer.allocateDirect(numChannels * numBoxes).order(ByteOrder.nativeOrder())

        postProcessor = YoloPostProcessor(
            numChannels, numBoxes, inputSize, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )

        val expected = 5 + YoloPostProcessor.KEYPOINT_COUNT * 3
        if (numChannels != expected) {
            Log.w(TAG, "$MODEL_FILE has $numChannels channels, expected $expected for a " +
                "${YoloPostProcessor.KEYPOINT_COUNT}-keypoint pose model - decode will be wrong")
        }

        val layout = if (channelsFirst) "NCHW" else "NHWC"
        Log.i(TAG, "Model loaded: input ${inputSize}x$inputSize $inputType $layout, output ${numChannels}x$numBoxes $outputType")
    }

    private fun createCpuInterpreter(): Interpreter =
        Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))

    /**
     * Times CPU against NNAPI on a real frame and keeps whichever is faster.
     *
     * Which delegate wins is a property of the device, the driver and the model,
     * not something a threshold can predict: on a RealWear T21G, NNAPI accepts
     * this graph and runs it correctly, but XNNPACK on 4 CPU threads is measurably
     * faster. Measuring costs about a second once, and stays right when the model
     * or the hardware changes.
     */
    private fun chooseDelegate() {
        val cpuMs = timeInference()
        val delegate = try {
            NnApiDelegate()
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI unavailable, staying on CPU", e)
            return
        }

        val cpuInterpreter = interpreter
        val nnApiMs = try {
            interpreter = Interpreter(modelBuffer, Interpreter.Options().addDelegate(delegate))
            timeInference()
        } catch (e: Exception) {
            // A driver can construct and then fail at run time. Never let that
            // reach the analysis thread - fall back and keep detecting.
            Log.w(TAG, "NNAPI failed at run time, staying on CPU", e)
            interpreter = cpuInterpreter
            delegate.close()
            return
        }

        if (nnApiMs < cpuMs) {
            Log.i(TAG, "Using NNAPI delegate (${nnApiMs}ms vs ${cpuMs}ms on CPU)")
            nnApiDelegate = delegate
            cpuInterpreter.close()
        } else {
            Log.i(TAG, "Using 4-thread CPU (${cpuMs}ms vs ${nnApiMs}ms on NNAPI)")
            interpreter.close()
            delegate.close()
            interpreter = cpuInterpreter
        }
    }

    /** Fastest of [BENCHMARK_RUNS] inferences on the frame already in [inputBuffer]. */
    private fun timeInference(): Long {
        var best = Long.MAX_VALUE
        repeat(BENCHMARK_RUNS) {
            val start = System.nanoTime()
            runInference()
            best = minOf(best, (System.nanoTime() - start) / 1_000_000)
        }
        return best
    }

    /** [bitmap] must already be square; it is scaled to the model input size. */
    fun detect(bitmap: Bitmap): List<Pose> {
        val scaled = if (bitmap.width == inputSize && bitmap.height == inputSize) bitmap
        else Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.rewind()
        if (channelsFirst) {
            for (shift in RGB_SHIFTS) {
                for (p in pixels) putSample(((p shr shift) and 0xFF) / 255f)
            }
        } else {
            for (p in pixels) {
                for (shift in RGB_SHIFTS) putSample(((p shr shift) and 0xFF) / 255f)
            }
        }

        if (!benchmarked) {
            benchmarked = true
            chooseDelegate()
        }
        return postProcessor.process(runInference())
    }

    private fun putSample(v: Float) {
        if (inputType == DataType.FLOAT32) inputBuffer.putFloat(v) else inputBuffer.put(quantize(v))
    }

    private fun quantize(v: Float): Byte =
        (v / inputScale + inputZeroPoint).toInt().coerceIn(-128, 127).toByte()

    /**
     * Returns output as [numChannels][numBoxes] floats, dequantized if needed.
     * The returned array is [outputArray] and is overwritten on the next call.
     */
    private fun runInference(): Array<FloatArray> {
        inputBuffer.rewind()
        if (outputType == DataType.FLOAT32) {
            interpreter.run(inputBuffer, outputHolder)
            return outputArray
        }
        val byteOut = quantizedOutput!!
        byteOut.rewind()
        interpreter.run(inputBuffer, byteOut)
        byteOut.rewind()
        for (c in 0 until numChannels) {
            for (b in 0 until numBoxes) {
                outputArray[c][b] = (byteOut.get().toInt() - outputZeroPoint) * outputScale
            }
        }
        return outputArray
    }

    fun close() {
        interpreter.close()
        nnApiDelegate?.close()
    }
}

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
        const val IOU_THRESHOLD = 0.45f
        private const val BENCHMARK_RUNS = 3
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
    // Starts on CPU so tensor shapes can be read without committing to a delegate;
    // the first detect() benchmarks both and keeps the winner. See chooseDelegate().
    private var interpreter: Interpreter = createCpuInterpreter()
    private var benchmarked = false

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
    private val postProcessor: YoloPostProcessor

    // Reused every frame. interpreter.run overwrites outputArray wholesale, and
    // quantizedOutput is rewound before each run, so neither needs reallocating.
    private val outputArray: Array<FloatArray>
    private val outputHolder: Array<Array<FloatArray>>
    private val quantizedOutput: ByteBuffer?

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

        outputArray = Array(numChannels) { FloatArray(numBoxes) }
        outputHolder = arrayOf(outputArray)
        quantizedOutput =
            if (outputType == DataType.FLOAT32) null
            else ByteBuffer.allocateDirect(numChannels * numBoxes).order(ByteOrder.nativeOrder())

        postProcessor = YoloPostProcessor(
            numChannels, numBoxes, inputSize, labels, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )

        val expectedLabels = numChannels - 4
        if (labels.size != expectedLabels) {
            Log.w(TAG, "$LABELS_FILE has ${labels.size} labels but the model has $expectedLabels " +
                "classes - detections will be mislabelled")
        }

        Log.i(TAG, "Model loaded: input ${inputSize}x$inputSize $inputType, output ${numChannels}x$numBoxes $outputType")
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

        if (!benchmarked) {
            benchmarked = true
            chooseDelegate()
        }
        return postProcessor.process(runInference())
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

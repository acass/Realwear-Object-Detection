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
import java.nio.FloatBuffer

/**
 * One instance mask ready to draw: an ALPHA_8 bitmap at prototype resolution,
 * and the normalized 0..1 rect over the square model input it belongs in.
 *
 * The bitmap is built on the analysis thread and never written again, so it
 * crosses to the UI thread immutable.
 */
data class InstanceMask(val bitmap: Bitmap, val box: RectF)

/**
 * [classId] is the model's class index, kept so the overlay can colour by class;
 * [label] is the same thing resolved through labels.txt.
 *
 * [boxIndex] is the detection's column in the raw output tensor. It survives NMS
 * so mask coefficients can be read straight out of the tensor afterwards, for
 * the few detections that are actually drawn, without copying 32 floats per
 * candidate on the way through.
 */
data class Detection(
    val box: RectF,
    val label: String,
    val confidence: Float,
    val classId: Int = -1,
    val mask: InstanceMask? = null,
    val boxIndex: Int = -1,
)

/**
 * Runs YOLOv8-seg TFLite inference and synthesizes an instance mask for each
 * detection it returns. Box coordinates in [Detection.box] are normalized 0..1
 * relative to the square model input.
 *
 * Tensors are channels-first: Ultralytics builds TFLite through LiteRT from
 * PyTorch, so the input is `[1, 3, H, W]` and the mask prototypes are
 * `[1, 32, H/4, W/4]` - not the channels-last layout older TFLite exports used.
 * Both are read from the model at startup rather than assumed; see init.
 *
 * The bundled asset is a float32 export at 256x256. Both quantized alternatives
 * were measured and rejected: full int8 collapses the segmentation head's
 * accuracy (COCO scores fell from 0.85 to 0.50, and a class was lost), and
 * `w8a32` - int8 weights with float32 activations - keeps that accuracy but will
 * not load, because TFLite 2.16.1's TRANSPOSE_CONV kernel requires its weights
 * and input to share a type. The prototype branch upsamples through
 * TRANSPOSE_CONV, so that rules it out until the runtime is newer.
 */
class Detector(context: Context) {

    companion object {
        private const val TAG = "Detector"
        private const val MODEL_FILE = "yolov8n_seg.tflite"
        private const val LABELS_FILE = "labels.txt"
        const val CONFIDENCE_THRESHOLD = 0.5f
        const val IOU_THRESHOLD = 0.45f
        const val MASK_THRESHOLD = 0.5f

        /**
         * Masks are only synthesized for the detections the overlay will draw.
         * Synthesizing one the user never sees is pure cost.
         */
        const val MAX_DETECTIONS = 5
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
    private val numChannels: Int   // 116 for COCO seg (4 box + 80 classes + 32 coeffs)
    private val numBoxes: Int
    private val numClasses: Int
    private val maskCoeffs: Int
    private val coeffStart: Int

    private val predictionIndex: Int
    private val protoIndex: Int
    private val protoC: Int
    private val protoH: Int
    private val protoW: Int

    private val inputBuffer: ByteBuffer
    private val pixels: IntArray
    private val postProcessor: YoloPostProcessor
    private val maskDecoder: MaskDecoder

    // Reused every frame. The interpreter overwrites both wholesale on each run.
    private val outputArray: Array<FloatArray>
    private val outputHolder: Array<Array<FloatArray>>
    private val protoBytes: ByteBuffer
    private val protoBuffer: FloatBuffer

    // Scratch, analysis-thread only: never handed to the UI thread.
    private val coeffScratch: FloatArray

    init {
        val inTensor = interpreter.getInputTensor(0)
        val inShape = inTensor.shape()          // [1, 3, H, W]
        require(inShape.size == 4 && inShape[1] == 3) {
            "Expected a channels-first [1, 3, H, W] input, got ${inShape.contentToString()}"
        }
        require(inShape[2] == inShape[3]) {
            "Expected a square input, got ${inShape[2]}x${inShape[3]}"
        }
        require(inTensor.dataType() == DataType.FLOAT32) {
            "Expected a float32 input, got ${inTensor.dataType()}"
        }
        inputSize = inShape[2]

        // Identify the outputs by rank rather than by index: the predictions are
        // rank 3 and the prototypes rank 4, which stays true regardless of what
        // order a given Ultralytics version happens to emit them in.
        var predIdx = -1
        var protoIdx = -1
        for (i in 0 until interpreter.outputTensorCount) {
            val t = interpreter.getOutputTensor(i)
            require(t.dataType() == DataType.FLOAT32) {
                "Output $i is ${t.dataType()}, expected float32 - export with quantize='w8a32'"
            }
            when (t.shape().size) {
                3 -> predIdx = i
                4 -> protoIdx = i
            }
            Log.i(TAG, "out[$i] ${t.shape().contentToString()} ${t.dataType()}")
        }
        require(predIdx >= 0 && protoIdx >= 0) {
            "Need a rank-3 prediction tensor and a rank-4 prototype tensor; this model has " +
                (0 until interpreter.outputTensorCount).joinToString {
                    interpreter.getOutputTensor(it).shape().contentToString()
                } + ". Is this a detection-only export rather than a -seg one?"
        }
        predictionIndex = predIdx
        protoIndex = protoIdx

        val predShape = interpreter.getOutputTensor(predIdx).shape()   // [1, 116, 2100]
        numChannels = predShape[1]
        numBoxes = predShape[2]

        val protoShape = interpreter.getOutputTensor(protoIdx).shape() // [1, 32, 80, 80]
        protoC = protoShape[1]
        protoH = protoShape[2]
        protoW = protoShape[3]

        maskCoeffs = protoC
        numClasses = numChannels - 4 - maskCoeffs
        coeffStart = 4 + numClasses

        // Throw rather than warn. A mismatch here does not produce an obvious
        // error downstream - it produces plausible-looking nonsense, mislabelled
        // detections and masks built from the wrong channels. Fail at startup,
        // where the message can still say what is wrong.
        require(numClasses == labels.size) {
            "$LABELS_FILE has ${labels.size} labels but the model has $numClasses classes " +
                "($numChannels channels - 4 box - $maskCoeffs mask coefficients)"
        }

        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)

        outputArray = Array(numChannels) { FloatArray(numBoxes) }
        outputHolder = arrayOf(outputArray)
        protoBytes = ByteBuffer.allocateDirect(protoC * protoH * protoW * 4)
            .order(ByteOrder.nativeOrder())
        protoBuffer = protoBytes.asFloatBuffer()

        coeffScratch = FloatArray(maskCoeffs)

        postProcessor = YoloPostProcessor(
            numClasses, numBoxes, inputSize, labels, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )
        maskDecoder = MaskDecoder(protoW, protoH, protoC, MASK_THRESHOLD)

        Log.i(
            TAG,
            "Model loaded: input ${inputSize}x$inputSize, $numClasses classes, " +
                "$numBoxes boxes, ${protoC}x${protoH}x$protoW prototypes"
        )
    }

    private fun createCpuInterpreter(): Interpreter =
        Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))

    /**
     * Times CPU against NNAPI on a real frame and keeps whichever is faster.
     *
     * Which delegate wins is a property of the device, the driver and the model,
     * not something a threshold can predict: on a RealWear T21G, NNAPI accepts
     * the detection graph and runs it correctly, but XNNPACK on 4 CPU threads is
     * measurably faster. Measuring costs about a second once, and stays right
     * when the model or the hardware changes.
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

    /**
     * [bitmap] must already be square; it is scaled to the model input size.
     *
     * When [targetLabel] is set only detections of that class are returned - the
     * filter runs before the [MAX_DETECTIONS] cap so a guided procedure's target
     * still gets a mask in a crowded scene, where it might not be among the five
     * strongest detections overall.
     */
    fun detect(bitmap: Bitmap, targetLabel: String? = null): List<Detection> {
        val scaled = if (bitmap.width == inputSize && bitmap.height == inputSize) bitmap
        else Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Channels-first: a whole red plane, then green, then blue.
        inputBuffer.rewind()
        for (p in pixels) inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
        for (p in pixels) inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
        for (p in pixels) inputBuffer.putFloat((p and 0xFF) / 255f)

        if (!benchmarked) {
            benchmarked = true
            chooseDelegate()
        }

        runInference()

        val found = postProcessor.process(outputArray, targetLabel, MAX_DETECTIONS)

        // A fresh list of fresh detections, each holding a bitmap this thread
        // will never touch again. Nothing here is reused across frames: a pooled
        // mask buffer rewritten by the next frame would tear under the UI thread
        // mid-draw, and at these sizes the allocation is not worth that risk.
        val result = ArrayList<Detection>(found.size)
        for (d in found) {
            for (k in 0 until maskCoeffs) coeffScratch[k] = outputArray[coeffStart + k][d.boxIndex]
            val bits = maskDecoder.decode(protoBuffer, coeffScratch, d.box)
            result.add(if (bits == null) d else d.copy(mask = InstanceMask(toBitmap(bits), bits.box)))
        }
        return result
    }

    /**
     * ALPHA_8, so the mask carries coverage only and the overlay's paint supplies
     * the colour. Rows are copied one at a time because a bitmap's rowBytes is
     * padded for alignment and is not always its width.
     */
    private fun toBitmap(m: MaskBits): Bitmap {
        val bmp = Bitmap.createBitmap(m.width, m.height, Bitmap.Config.ALPHA_8)
        val stride = bmp.rowBytes
        val buf = ByteBuffer.allocate(stride * m.height)
        for (y in 0 until m.height) {
            buf.position(y * stride)
            buf.put(m.alpha, y * m.width, m.width)
        }
        buf.rewind()
        bmp.copyPixelsFromBuffer(buf)
        return bmp
    }

    /**
     * Fills [outputArray] with the predictions and [protoBuffer] with the mask
     * prototypes. Both are reused, and both are overwritten on the next call.
     */
    private fun runInference() {
        inputBuffer.rewind()
        protoBytes.rewind()
        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(inputBuffer),
            mapOf(predictionIndex to outputHolder, protoIndex to protoBytes),
        )
    }

    fun close() {
        interpreter.close()
        nnApiDelegate?.close()
    }
}

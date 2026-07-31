package com.crossmedia.objectdetect

import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.MappedByteBuffer

/** The winning interpreter and the delegate that must outlive it. */
class DelegateChoice(
    val interpreter: Interpreter,
    val delegate: AutoCloseable?,
    val name: String,
    val ms: Long,
)

/**
 * Picks the fastest available TFLite delegate for a model by timing all of them.
 *
 * Which one wins is a property of the device, the driver and the model, not something
 * a threshold can predict: on the Navigator 500 the detector is fastest on 4-thread
 * XNNPACK while the much heavier depth graph is nearly 3x faster on the Adreno GPU.
 * NNAPI is measured too and consistently loses, which matches the device shipping no
 * NNAPI vendor driver -- it is running a CPU reference path.
 */
object DelegateRace {

    /**
     * @param cpu a working CPU interpreter, already primed with input. Closed by this
     *   function if something else wins.
     * @param time runs inference on the given interpreter and returns its best time.
     */
    fun run(
        tag: String,
        modelBuffer: MappedByteBuffer,
        cpu: Interpreter,
        time: (Interpreter) -> Long,
    ): DelegateChoice {
        // Tracks the current winner, not the CPU baseline. With more than one
        // challenger those differ: once GPU has won, a slower NNAPI must fall back to
        // GPU, and falling back to CPU instead silently discards the win.
        var bestInterpreter = cpu
        var bestDelegate: AutoCloseable? = null
        var bestName = "CPU(4t)"
        var bestMs = time(cpu)
        Log.i(tag, "CPU(4t): ${bestMs}ms")

        for ((name, make) in listOf<Pair<String, () -> Delegate>>(
            "GPU" to { GpuDelegate() },
            "NNAPI" to { NnApiDelegate() },
        )) {
            // Throwable, not Exception: a delegate whose optional artifact is missing
            // fails with NoClassDefFoundError, and missing natives raise
            // UnsatisfiedLinkError. Both are Errors, and an unavailable accelerator
            // must degrade to the current best, never kill the analysis thread.
            val delegate = try {
                make()
            } catch (e: Throwable) {
                Log.w(tag, "$name delegate unavailable", e)
                continue
            }
            val candidate = try {
                Interpreter(modelBuffer, Interpreter.Options().addDelegate(delegate))
            } catch (e: Throwable) {
                Log.w(tag, "$name interpreter would not build", e)
                (delegate as AutoCloseable).close()
                continue
            }
            val ms = try {
                time(candidate)
            } catch (e: Throwable) {
                // A driver can construct and then fail at run time.
                Log.w(tag, "$name failed at run time", e)
                candidate.close()
                (delegate as AutoCloseable).close()
                continue
            }
            Log.i(tag, "$name: ${ms}ms")

            if (ms < bestMs) {
                bestInterpreter.close()
                bestDelegate?.close()
                bestInterpreter = candidate
                bestDelegate = delegate as AutoCloseable
                bestName = name
                bestMs = ms
            } else {
                candidate.close()
                (delegate as AutoCloseable).close()
            }
        }

        Log.i(tag, "Using $bestName (${bestMs}ms)")
        return DelegateChoice(bestInterpreter, bestDelegate, bestName, bestMs)
    }
}

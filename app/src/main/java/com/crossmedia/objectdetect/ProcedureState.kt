package com.crossmedia.objectdetect

/**
 * Which screen the guided procedure is showing.
 *
 * [IDLE] is the app's original free-running detection; the procedure only owns
 * the UI in [RUNNING] and [SUMMARY].
 */
enum class Mode { IDLE, RUNNING, SUMMARY }

/**
 * One step's outcome.
 *
 * The operator's verdict and the detector's are recorded separately on purpose:
 * the point of the demo is comparing them. [foundByOperator] is null until the
 * step is answered; [peakConfidence] is 0f until the detector sees [label] at
 * all, which is an unambiguous sentinel because every detection reaching here
 * has already cleared [Detector.CONFIDENCE_THRESHOLD].
 */
data class StepResult(
    val label: String,
    var foundByOperator: Boolean? = null,
    var peakConfidence: Float = 0f,
)

/**
 * The 5-step guided procedure: which object to look for, what the operator said
 * about it, and the strongest the detector ever saw it.
 *
 * Deliberately free of Android imports so the step arithmetic can be checked on
 * the desktop JVM, the same reason [YoloPostProcessor] is split out of
 * [Detector]. Confined to the UI thread; the analysis thread reaches it only
 * through [MainActivity]'s volatile target label and a posted [observe].
 */
class ProcedureState(targets: List<String>) {

    companion object {
        /**
         * COCO classes for the demo, in step order. Chosen for hit rate at
         * 320x320: no `mouse` or `scissors` (too small to survive the 0.5
         * threshold), no `book` (weak class), no `person` (always in frame).
         * `cup` rather than "coffee cup" — COCO has no such class.
         */
        val DEFAULT_TARGETS = listOf("cup", "keyboard", "laptop", "bottle", "cell phone")
    }

    val results: List<StepResult> = targets.map { StepResult(it) }

    var mode: Mode = Mode.IDLE
        private set

    var stepIndex: Int = 0
        private set

    val currentTarget: String?
        get() = if (mode == Mode.RUNNING) results[stepIndex].label else null

    val foundCount: Int
        get() = results.count { it.foundByOperator == true }

    /** Enters step 1, discarding any previous run's verdicts and measurements. */
    fun start() {
        for (r in results) {
            r.foundByOperator = null
            r.peakConfidence = 0f
        }
        stepIndex = 0
        mode = Mode.RUNNING
    }

    /**
     * Records the operator's verdict for the current step and advances. Answering
     * the last step ends the run at [Mode.SUMMARY] rather than a sixth step.
     */
    fun answer(found: Boolean) {
        if (mode != Mode.RUNNING) return
        results[stepIndex].foundByOperator = found
        if (stepIndex == results.lastIndex) {
            mode = Mode.SUMMARY
        } else {
            stepIndex++
        }
    }

    /**
     * Returns to the previous step, clearing both its verdict and its measurement.
     *
     * The measurement has to go too: a re-run of the step is a fresh look at the
     * object, so keeping the old peak would credit the detector for a frame the
     * operator has just disowned. No-op on step 1 — there is nothing to undo.
     */
    fun goBack() {
        if (mode != Mode.RUNNING || stepIndex == 0) return
        stepIndex--
        results[stepIndex].foundByOperator = null
        results[stepIndex].peakConfidence = 0f
    }

    /**
     * Reports the strongest detection of the current target in one frame, keeping
     * the running maximum. Called once per analysed frame, including frames where
     * the target was absent (0f), which harmlessly leaves the peak alone.
     */
    fun observe(confidence: Float) {
        if (mode != Mode.RUNNING) return
        val r = results[stepIndex]
        if (confidence > r.peakConfidence) r.peakConfidence = confidence
    }

    /** Leaves the summary and hands the screen back to free-running detection. */
    fun exit() {
        mode = Mode.IDLE
        stepIndex = 0
    }
}

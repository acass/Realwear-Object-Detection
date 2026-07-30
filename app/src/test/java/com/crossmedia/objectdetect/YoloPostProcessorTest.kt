package com.crossmedia.objectdetect

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Small synthetic tensors rather than the real 84x8400: the decode is per-box,
 * so 2 classes and a handful of boxes exercise every branch.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class YoloPostProcessorTest {

    private val labels = listOf("person", "bicycle")
    private val numClasses = 2
    private val channels = 6   // 4 box + 2 classes
    private val inputSize = 640

    private fun processor(boxes: Int, labels: List<String> = this.labels) =
        YoloPostProcessor(numClasses, boxes, inputSize, labels, 0.5f, 0.45f)

    /** [boxes] is (cx, cy, w, h, score0, score1) per box. */
    private fun tensor(vararg boxes: FloatArray): Array<FloatArray> {
        val out = Array(channels) { FloatArray(boxes.size) }
        boxes.forEachIndexed { b, values ->
            for (c in 0 until channels) out[c][b] = values[c]
        }
        return out
    }

    @Test
    fun `decodes a single box to centre-form rect and label`() {
        val out = tensor(floatArrayOf(0.5f, 0.5f, 0.2f, 0.4f, 0.9f, 0.1f))
        val result = processor(1).process(out)

        assertEquals(1, result.size)
        val d = result[0]
        assertEquals("person", d.label)
        assertEquals(0.9f, d.confidence, 1e-6f)
        assertEquals(0.4f, d.box.left, 1e-6f)     // cx - w/2
        assertEquals(0.3f, d.box.top, 1e-6f)      // cy - h/2
        assertEquals(0.6f, d.box.right, 1e-6f)
        assertEquals(0.7f, d.box.bottom, 1e-6f)
    }

    @Test
    fun `caps at the strongest detections, in descending confidence`() {
        // Eight disjoint boxes so NMS keeps them all, ascending confidence.
        val boxes = (1..8).map { i ->
            floatArrayOf(0.06f * i, 0.5f, 0.05f, 0.05f, 0.5f + 0.05f * i, 0f)
        }.toTypedArray()
        val result = processor(8).process(tensor(*boxes), maxDetections = 5)

        assertEquals(5, result.size)
        val confidences = result.map { it.confidence }
        assertEquals(0.9f, confidences.first(), 1e-6f)
        assertTrue("must be descending", confidences.zipWithNext().all { (a, b) -> a >= b })
    }

    @Test
    fun `an uncapped call keeps everything NMS kept`() {
        val boxes = (1..8).map { i ->
            floatArrayOf(0.06f * i, 0.5f, 0.05f, 0.05f, 0.5f + 0.05f * i, 0f)
        }.toTypedArray()

        assertEquals(8, processor(8).process(tensor(*boxes)).size)
    }

    @Test
    fun `a target filter keeps only that class`() {
        val out = tensor(
            floatArrayOf(0.2f, 0.5f, 0.05f, 0.05f, 0.9f, 0.1f),   // person
            floatArrayOf(0.8f, 0.5f, 0.05f, 0.05f, 0.1f, 0.8f),   // bicycle
        )
        val result = processor(2).process(out, targetLabel = "bicycle")

        assertEquals(1, result.size)
        assertEquals("bicycle", result[0].label)
    }

    @Test
    fun `the target survives the cap even when it is not the strongest`() {
        // Six strong people and one weak bicycle: an unfiltered top-5 would drop
        // the bicycle, so a procedure step targeting it would show no mask.
        val boxes = (1..6).map { i ->
            floatArrayOf(0.06f * i, 0.2f, 0.05f, 0.05f, 0.9f, 0f)
        } + listOf(floatArrayOf(0.5f, 0.8f, 0.05f, 0.05f, 0f, 0.55f))
        val result = processor(7).process(
            tensor(*boxes.toTypedArray()), targetLabel = "bicycle", maxDetections = 5
        )

        assertEquals(1, result.size)
        assertEquals("bicycle", result[0].label)
    }

    @Test
    fun `ignores channels past the class scores`() {
        // 4 box + 2 classes + 2 trailing channels, as a segmentation export has:
        // mask coefficients are unbounded and would win an unguarded argmax.
        val out = Array(8) { FloatArray(1) }
        val values = floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.1f, 9f, -9f)
        for (c in 0 until 8) out[c][0] = values[c]

        val result = processor(1).process(out)

        assertEquals(1, result.size)
        assertEquals("person", result[0].label)
        assertEquals(0, result[0].classId)
        assertEquals(0.6f, result[0].confidence, 1e-6f)
    }

    @Test
    fun `carries class id and box index through to the detection`() {
        val out = tensor(
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.1f),   // below threshold, dropped
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.9f),   // bicycle, tensor column 1
        )
        val result = processor(2).process(out)

        assertEquals(1, result.size)
        assertEquals(1, result[0].classId)
        assertEquals(1, result[0].boxIndex)
    }

    @Test
    fun `picks the highest scoring class channel`() {
        val out = tensor(floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.6f, 0.8f))
        val result = processor(1).process(out)

        assertEquals(1, result.size)
        assertEquals("bicycle", result[0].label)
        assertEquals(0.8f, result[0].confidence, 1e-6f)
    }

    @Test
    fun `drops boxes below the confidence threshold`() {
        val out = tensor(
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.49f, 0.1f),   // just under
            floatArrayOf(0.1f, 0.1f, 0.05f, 0.05f, 0.51f, 0.1f), // just over
        )
        val result = processor(2).process(out)

        assertEquals(1, result.size)
        assertEquals(0.51f, result[0].confidence, 1e-6f)
    }

    @Test
    fun `suppresses an overlapping box and keeps the stronger one`() {
        // Same 0.4-square shifted by 0.02: IoU well above 0.45.
        val out = tensor(
            floatArrayOf(0.50f, 0.50f, 0.4f, 0.4f, 0.7f, 0.0f),
            floatArrayOf(0.52f, 0.50f, 0.4f, 0.4f, 0.9f, 0.0f),
        )
        val result = processor(2).process(out)

        assertEquals(1, result.size)
        assertEquals(0.9f, result[0].confidence, 1e-6f)
    }

    @Test
    fun `keeps disjoint boxes`() {
        val out = tensor(
            floatArrayOf(0.2f, 0.2f, 0.1f, 0.1f, 0.7f, 0.0f),
            floatArrayOf(0.8f, 0.8f, 0.1f, 0.1f, 0.9f, 0.0f),
        )
        val result = processor(2).process(out)

        assertEquals(2, result.size)
    }

    @Test
    fun `divides by input size when the model emits pixel-space coordinates`() {
        // w of 128 trips the coordMax > 1.5 guard; everything scales by inputSize.
        val out = tensor(floatArrayOf(320f, 320f, 128f, 64f, 0.9f, 0.0f))
        val result = processor(1).process(out)

        assertEquals(1, result.size)
        val box = result[0].box
        assertEquals(0.4f, box.left, 1e-5f)      // (320 - 64) / 640
        assertEquals(0.45f, box.top, 1e-5f)      // (320 - 32) / 640
        assertEquals(0.6f, box.right, 1e-5f)
        assertEquals(0.55f, box.bottom, 1e-5f)
    }

    @Test
    fun `falls back to a synthetic label when the label file is short`() {
        val out = tensor(floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.9f))
        val result = processor(1, labels = listOf("person")).process(out)

        assertEquals(1, result.size)
        assertEquals("class 1", result[0].label)
    }

    @Test
    fun `returns nothing when every box is below threshold`() {
        val out = tensor(
            floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.2f),
            floatArrayOf(0.3f, 0.3f, 0.2f, 0.2f, 0.0f, 0.0f),
        )
        assertTrue(processor(2).process(out).isEmpty())
    }
}

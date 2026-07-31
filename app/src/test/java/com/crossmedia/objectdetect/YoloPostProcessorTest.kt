package com.crossmedia.objectdetect

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Small synthetic tensors rather than the real 56x2100: the decode is per-box,
 * so a handful of boxes exercises every branch.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class YoloPostProcessorTest {

    private val channels = 5 + YoloPostProcessor.KEYPOINT_COUNT * 3   // 56
    private val inputSize = 320

    private fun processor(boxes: Int) =
        YoloPostProcessor(channels, boxes, inputSize, 0.5f, 0.45f)

    /**
     * One box: cx, cy, w, h, person score, then 17 x (x, y, conf).
     * Keypoints default to [kpt], which keeps the callers readable when only
     * the box matters.
     */
    private fun box(
        cx: Float, cy: Float, w: Float, h: Float, score: Float, kpt: FloatArray? = null,
    ): FloatArray {
        val values = FloatArray(channels)
        values[0] = cx; values[1] = cy; values[2] = w; values[3] = h; values[4] = score
        kpt?.copyInto(values, 5)
        return values
    }

    private fun tensor(vararg boxes: FloatArray): Array<FloatArray> {
        val out = Array(channels) { FloatArray(boxes.size) }
        boxes.forEachIndexed { b, values ->
            for (c in 0 until channels) out[c][b] = values[c]
        }
        return out
    }

    @Test
    fun `decodes person score and every keypoint`() {
        // Keypoint i sits at (i/100, i/50) with confidence 1 - i/100.
        val kpt = FloatArray(YoloPostProcessor.KEYPOINT_COUNT * 3)
        for (i in 0 until YoloPostProcessor.KEYPOINT_COUNT) {
            kpt[i * 3] = i / 100f
            kpt[i * 3 + 1] = i / 50f
            kpt[i * 3 + 2] = 1f - i / 100f
        }
        val result = processor(1).process(tensor(box(0.5f, 0.5f, 0.2f, 0.4f, 0.9f, kpt)))

        assertEquals(1, result.size)
        val pose = result[0]
        assertEquals(0.9f, pose.score, 1e-6f)
        assertEquals(YoloPostProcessor.KEYPOINT_COUNT * 3, pose.keypoints.size)
        for (i in 0 until YoloPostProcessor.KEYPOINT_COUNT) {
            assertEquals(i / 100f, pose.keypoints[i * 3], 1e-6f)
            assertEquals(i / 50f, pose.keypoints[i * 3 + 1], 1e-6f)
            assertEquals(1f - i / 100f, pose.keypoints[i * 3 + 2], 1e-6f)
        }
    }

    @Test
    fun `drops people below the confidence threshold`() {
        val result = processor(2).process(
            tensor(
                box(0.5f, 0.5f, 0.2f, 0.2f, 0.49f),   // just under
                box(0.1f, 0.1f, 0.05f, 0.05f, 0.51f), // just over
            )
        )

        assertEquals(1, result.size)
        assertEquals(0.51f, result[0].score, 1e-6f)
    }

    @Test
    fun `suppresses an overlapping person and keeps the stronger one`() {
        // Same 0.4-square shifted by 0.02: IoU well above 0.45.
        val result = processor(2).process(
            tensor(
                box(0.50f, 0.50f, 0.4f, 0.4f, 0.7f),
                box(0.52f, 0.50f, 0.4f, 0.4f, 0.9f),
            )
        )

        assertEquals(1, result.size)
        assertEquals(0.9f, result[0].score, 1e-6f)
    }

    @Test
    fun `keeps people who do not overlap`() {
        val result = processor(2).process(
            tensor(
                box(0.2f, 0.2f, 0.1f, 0.1f, 0.7f),
                box(0.8f, 0.8f, 0.1f, 0.1f, 0.9f),
            )
        )

        assertEquals(2, result.size)
    }

    @Test
    fun `divides keypoints by input size when the model emits pixel-space coordinates`() {
        // Keypoint 0 at (160, 80) px; the rest stay at the origin.
        val kpt = FloatArray(YoloPostProcessor.KEYPOINT_COUNT * 3)
        kpt[0] = 160f; kpt[1] = 80f; kpt[2] = 0.9f
        val result = processor(1).process(tensor(box(160f, 160f, 64f, 32f, 0.9f, kpt)))

        assertEquals(1, result.size)
        val pose = result[0]
        assertEquals(0.5f, pose.keypoints[0], 1e-5f)    // 160 / 320
        assertEquals(0.25f, pose.keypoints[1], 1e-5f)   // 80 / 320
        assertEquals(0.9f, pose.keypoints[2], 1e-6f)    // confidence is not a coordinate
    }

    @Test
    fun `keeps normalized keypoints when the box is pixel-space`() {
        // Ultralytics exports have varied on this; the two spaces are guarded
        // independently so a mixed export still decodes.
        val kpt = FloatArray(YoloPostProcessor.KEYPOINT_COUNT * 3)
        kpt[0] = 0.5f; kpt[1] = 0.25f; kpt[2] = 0.9f
        val result = processor(1).process(tensor(box(160f, 160f, 64f, 32f, 0.9f, kpt)))

        assertEquals(0.5f, result[0].keypoints[0], 1e-5f)
        assertEquals(0.25f, result[0].keypoints[1], 1e-5f)
    }

    @Test
    fun `returns nothing when every person is below threshold`() {
        val out = tensor(
            box(0.5f, 0.5f, 0.2f, 0.2f, 0.1f),
            box(0.3f, 0.3f, 0.2f, 0.2f, 0.0f),
        )
        assertTrue(processor(2).process(out).isEmpty())
    }
}

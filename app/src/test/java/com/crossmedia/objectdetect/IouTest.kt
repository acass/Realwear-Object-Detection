package com.crossmedia.objectdetect

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class IouTest {

    private val p = YoloPostProcessor(56, 1, 320, 0.5f, 0.45f)

    private fun candidate(box: RectF, score: Float) =
        YoloPostProcessor.Candidate(box, Pose(FloatArray(YoloPostProcessor.KEYPOINT_COUNT * 3), score))

    @Test
    fun `identical boxes overlap completely`() {
        val a = RectF(0f, 0f, 1f, 1f)
        assertEquals(1f, p.iou(a, RectF(0f, 0f, 1f, 1f)), 1e-6f)
    }

    @Test
    fun `disjoint boxes do not overlap`() {
        assertEquals(0f, p.iou(RectF(0f, 0f, 1f, 1f), RectF(2f, 2f, 3f, 3f)), 1e-6f)
    }

    @Test
    fun `boxes sharing only an edge do not overlap`() {
        assertEquals(0f, p.iou(RectF(0f, 0f, 1f, 1f), RectF(1f, 0f, 2f, 1f)), 1e-6f)
    }

    @Test
    fun `half overlap is one third`() {
        // Intersection 0.5, union 1.5.
        val iou = p.iou(RectF(0f, 0f, 1f, 1f), RectF(0.5f, 0f, 1.5f, 1f))
        assertEquals(1f / 3f, iou, 1e-6f)
    }

    @Test
    fun `degenerate zero-area box yields zero rather than NaN`() {
        val iou = p.iou(RectF(0f, 0f, 1f, 1f), RectF(0.5f, 0.5f, 0.5f, 0.5f))
        assertFalse(iou.isNaN())
        assertEquals(0f, iou, 1e-6f)
    }

    @Test
    fun `nms keeps the strongest of a cluster and every isolated box`() {
        val cluster = listOf(
            candidate(RectF(0f, 0f, 1f, 1f), 0.6f),
            candidate(RectF(0.05f, 0.05f, 1.05f, 1.05f), 0.9f),
            candidate(RectF(5f, 5f, 6f, 6f), 0.7f),
        )
        val kept = p.nms(cluster)

        assertEquals(2, kept.size)
        assertEquals(0.9f, kept[0].pose.score, 1e-6f)
        assertEquals(0.7f, kept[1].pose.score, 1e-6f)
    }

    @Test
    fun `nms on an empty list returns empty`() {
        assertEquals(0, p.nms(emptyList()).size)
    }
}

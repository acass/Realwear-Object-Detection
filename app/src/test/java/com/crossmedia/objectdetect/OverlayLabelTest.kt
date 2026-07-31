package com.crossmedia.objectdetect

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

/**
 * The pill text is what the wearer actually reads, and it is the one part of the depth
 * feature that cannot be confirmed from logcat without a COCO object in front of the
 * camera. These pin it down without hardware.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OverlayLabelTest {

    private val view =
        OverlayView(RuntimeEnvironment.getApplication(), null)

    private fun detection(label: String, confidence: Float, metres: Float?) =
        Detection(RectF(0.4f, 0.4f, 0.6f, 0.6f), label, confidence, metres)

    @Test
    fun distanceReplacesConfidenceWhenPresent() {
        assertEquals("PERSON 2.3M", view.labelFor(detection("person", 0.87f, 2.34f)))
    }

    @Test
    fun fallsBackToConfidenceWhenDepthIsNull() {
        // Null is what DepthSampler returns beyond MAX_RANGE_METRES, so this is the
        // out-of-range callout, not just a missing-model case.
        assertEquals("PERSON 87%", view.labelFor(detection("person", 0.87f, null)))
    }

    @Test
    fun distanceRoundsToOneDecimal() {
        assertEquals("CHAIR 10.0M", view.labelFor(detection("chair", 0.6f, 9.96f)))
    }

    @Test
    fun subMetreDistanceKeepsLeadingZero() {
        assertEquals("CUP 0.4M", view.labelFor(detection("cup", 0.6f, 0.42f)))
    }

    @Test
    fun multiWordLabelIsUppercased() {
        assertEquals("CELL PHONE 1.5M", view.labelFor(detection("cell phone", 0.9f, 1.5f)))
    }
}

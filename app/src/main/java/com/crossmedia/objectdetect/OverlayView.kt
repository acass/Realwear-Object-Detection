package com.crossmedia.objectdetect

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Draws detection boxes over the camera preview. Detections use normalized
 * 0..1 coordinates over the square center-crop of the preview; this view is
 * laid out to exactly cover that crop region, so mapping is a simple scale.
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.GREEN
    }
    // Large text for the Navigator's small monocular display
    private val textPaint = Paint().apply {
        color = Color.GREEN
        textSize = 48f * resources.displayMetrics.scaledDensity
        isFakeBoldText = true
    }
    private val textBgPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
    }

    fun setDetections(list: List<Detection>) {
        detections = list
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        for (d in detections) {
            val r = RectF(d.box.left * w, d.box.top * h, d.box.right * w, d.box.bottom * h)
            canvas.drawRect(r, boxPaint)

            val label = "${d.label} ${(d.confidence * 100).toInt()}%"
            val textW = textPaint.measureText(label)
            val textH = textPaint.textSize
            val ty = (r.top - 8f).coerceAtLeast(textH)
            canvas.drawRect(r.left, ty - textH, r.left + textW + 16f, ty + 12f, textBgPaint)
            canvas.drawText(label, r.left + 8f, ty, textPaint)
        }
    }
}

package com.app.realtimeobjectdetection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color.*
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.app.realtimeobjectdetection.databinding.ActivityMainBinding
import com.app.realtimeobjectdetection.ml.MetadataModel
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraDevice: CameraDevice
    private lateinit var model: MetadataModel
    private val paint = Paint()
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var bitmap: Bitmap
    private val colors = listOf(
        BLUE, GREEN, RED, CYAN, GRAY, BLACK,
        DKGRAY, MAGENTA, YELLOW,
    )
    private lateinit var labels: List<String>
    private val handlerThread = HandlerThread("videoThread").apply { start() }
    private val handler = Handler(handlerThread.looper)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        getCameraPermission()
        setupValues()
    }

    private fun setupValues() {
        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = MetadataModel.newInstance(this)

        binding.textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(
                surfaceTexture: SurfaceTexture,
                width: Int,
                height: Int,
            ) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(
                surfaceTexture: SurfaceTexture,
                width: Int,
                height: Int,
            ) {
            }

            override fun onSurfaceTextureDestroyed(surfaceTexture: SurfaceTexture) = false
            override fun onSurfaceTextureUpdated(surfaceTexture: SurfaceTexture) {
                processTextureView()
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private fun processTextureView() {
        bitmap = binding.textureView.bitmap!!
        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)
        val outputs = model.process(image)
        drawObjects(outputs)
    }

    private fun drawObjects(outputs: MetadataModel.Outputs) {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray

        val h = mutableBitmap.height
        val w = mutableBitmap.width
        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f
        var x: Int
        paint.style = Paint.Style.STROKE

        scores.forEachIndexed { index, fl ->
            x = index
            x *= NUM_COORDINATES
            if (fl > CONFIDENCE_THRESHOLD) {
                paint.color = colors[index]
                paint.style = Paint.Style.STROKE
                canvas.drawRect(
                    RectF(
                        locations[x + 1] * w,
                        locations[x] * h,
                        locations[x + 3] * w,
                        locations[x + 2] * h,
                    ),
                    paint,
                )
                paint.style = Paint.Style.FILL
                canvas.drawText(
                    labels[classes[index].toInt()] + " " + fl.toString(),
                    locations[x + 1] * w,
                    locations[x] * h,
                    paint,
                )
            }
        }

        binding.imageView.setImageBitmap(mutableBitmap)
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        try {
            val cameraId = cameraManager.cameraIdList[0]
            cameraManager.openCamera(cameraId, cameraStateCallback, handler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreviewSession()
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val surfaceTexture = binding.textureView.surfaceTexture
            val surface = Surface(surfaceTexture)
            val captureRequest =
                cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(surface)
                }
            cameraDevice.createCaptureSession(
                listOf(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(captureRequest.build(), null, null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {}
                },
                handler,
            )
        } catch (e: CameraAccessException) {
            Log.e("CameraPreviewSession", "$e")
        }
    }

    private fun getCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(android.Manifest.permission.CAMERA),
                PERMISSION_CAMERA_REQUEST_CODE,
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) getCameraPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    companion object {
        private const val PERMISSION_CAMERA_REQUEST_CODE = 101
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val NUM_COORDINATES = 4
    }
}

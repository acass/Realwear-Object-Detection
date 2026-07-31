plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.crossmedia.objectdetect"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.crossmedia.objectdetect"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // Keep the .tflite asset uncompressed so it can be memory-mapped
    androidResources {
        noCompress += "tflite"
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")

    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    // Adreno 610 on the Navigator's SM6115 exposes /vendor/lib64/libOpenCL.so, and the
    // device ships no NNAPI vendor driver (lshal lists none), so GPU is the only real
    // accelerator available. chooseDelegate() races it against CPU at startup.
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    // GpuDelegate's constructor references GpuDelegateFactory, which ships in this
    // separate artifact. Without it the class links only at construction time and
    // throws NoClassDefFoundError, not an Exception.
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.16.1")

    // Robolectric supplies real RectF/Canvas on the desktop JVM, so the detection
    // geometry is testable without a device. The TFLite interpreter is not — its
    // natives are Android-only, which is why the decode logic lives in
    // YoloPostProcessor rather than behind the Interpreter.
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
}

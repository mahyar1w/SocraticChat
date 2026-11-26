package com.socraticchat.app

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebSettings
import android.webkit.WebView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private lateinit var webView: WebView
    private val io = Executors.newCachedThreadPool()

    @Volatile
    private var engine: LlamaEngine? = null

    @Volatile
    private var teacherKeyMemory: String = ""

    private val pickModelReqCode = 1001

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = false
            settings.allowFileAccess = false
            settings.cacheMode = WebSettings.LOAD_NO_CACHE
            addJavascriptInterface(Bridge(), "AndroidBridge")
            loadUrl("file:///android_asset/index.html")
        }
        setContentView(webView)
    }

    override fun onDestroy() {
        engine?.stop()
        engine = null
        io.shutdownNow()
        super.onDestroy()
    }

    private fun evalJs(script: String) {
        runOnUiThread {
            webView.evaluateJavascript(script, null)
        }
    }

    private fun emitEvent(type: String, payload: JSONObject) {
        val wrapped = JSONObject().put("type", type).put("payload", payload)
        evalJs("window.onNativeEvent && window.onNativeEvent(${JSONObject.quote(wrapped.toString())});")
    }

    private fun startEngineAsync(preferredPath: String) {
        emitEvent("engine", JSONObject().put("status", "starting").put("detail", "Loading 0.8B model…"))
        io.execute {
            engine?.stop()
            engine = null

            val modelFile = LlamaEngine.findModel(this, preferredPath)
            if (modelFile == null) {
                emitEvent(
                    "engine",
                    JSONObject()
                        .put("status", "missing_model")
                        .put("detail", "Import Qwen3.5-0.8B-Socratic-Q8_0.gguf in Settings to start."),
                )
                return@execute
            }

            val res = LlamaEngine.start(this, modelFile)
            res.fold(
                onSuccess = { eng ->
                    engine = eng
                    emitEvent(
                        "engine",
                        JSONObject()
                            .put("status", "ready")
                            .put("detail", "Local engine ready")
                            .put("modelPath", modelFile.absolutePath),
                    )
                },
                onFailure = { err ->
                    emitEvent(
                        "engine",
                        JSONObject()
                            .put("status", "error")
                            .put("detail", err.message ?: "Failed to start llama-server"),
                    )
                },
            )
        }
    }

    @Deprecated("Using standard Activity result API for zero-dependency build")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == pickModelReqCode && resultCode == RESULT_OK) {
            val uri = data?.data ?: return
            importModelFromUri(uri)
        }
    }

    private fun importModelFromUri(uri: Uri) {
        emitEvent("engine", JSONObject().put("status", "starting").put("detail", "Importing GGUF model…"))
        io.execute {
            try {
                val dst = File(filesDir, LlamaEngine.DEFAULT_GGUF_NAME)
                val tmp = File(filesDir, "${LlamaEngine.DEFAULT_GGUF_NAME}.tmp")
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(tmp).use { output ->
                        val buf = ByteArray(1024 * 1024)
                        while (true) {
                            val n = input.read(buf)
                            if (n <= 0) break
                            output.write(buf, 0, n)
                        }
                    }
                } ?: throw IllegalStateException("Could not open selected file")

                if (dst.exists()) dst.delete()
                if (!tmp.renameTo(dst)) {
                    throw IllegalStateException("Failed saving imported model")
                }

                val state = StateStore.loadJson(this)
                state.optJSONObject("config")?.put("ggufPath", dst.absolutePath)
                StateStore.saveJson(this, state)
                emitEvent("model_imported", JSONObject().put("path", dst.absolutePath))
                startEngineAsync(dst.absolutePath)
            } catch (e: Exception) {
                emitEvent(
                    "engine",
                    JSONObject()
                        .put("status", "error")
                        .put("detail", "Model import failed: ${e.message}"),
                )
            }
        }
    }

    inner class Bridge {
        @JavascriptInterface
        fun loadState(): String {
            return StateStore.loadJson(this@MainActivity).toString()
        }

        @JavascriptInterface
        fun saveState(rawJson: String) {
            try {
                val obj = JSONObject(rawJson)
                StateStore.saveJson(this@MainActivity, obj)
            } catch (_: Exception) {
            }
        }

        @JavascriptInterface
        fun setTeacherKey(key: String) {
            teacherKeyMemory = key
        }

        @JavascriptInterface
        fun startEngine(ggufPath: String) {
            startEngineAsync(ggufPath)
        }

        @JavascriptInterface
        fun pickModel() {
            runOnUiThread {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }
                @Suppress("DEPRECATION")
                startActivityForResult(intent, pickModelReqCode)
            }
        }

        @JavascriptInterface
        fun copyText(text: String) {
            runOnUiThread {
                val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("Socratic Chat", text))
            }
        }

        @JavascriptInterface
        fun sendPrompt(convId: Long, historyJson: String, configJson: String) {
            val eng = engine
            if (eng == null) {
                emitEvent(
                    "reply",
                    JSONObject()
                        .put("convId", convId)
                        .put("ok", false)
                        .put("error", "Local llama.cpp engine is not ready yet."),
                )
                return
            }

            io.execute {
                try {
                    val arr = JSONArray(historyJson)
                    val history = mutableListOf<ChatMessage>()
                    for (i in 0 until arr.length()) {
                        val m = arr.getJSONObject(i)
                        history.add(ChatMessage(m.getString("role"), m.getString("content")))
                    }
                    val c = JSONObject(configJson)
                    val cfg = RequestConfig(
                        useTeacher = c.optBoolean("useTeacher", false),
                        teacherUrl = c.optString("teacherUrl", ""),
                        teacherModel = c.optString("teacherModel", ""),
                        teacherKey = teacherKeyMemory,
                        studentUrl = eng.baseUrl,
                    )
                    val reply = LlmClient.generateReply(history, cfg)
                    emitEvent(
                        "reply",
                        JSONObject()
                            .put("convId", convId)
                            .put("ok", true)
                            .put("text", reply),
                    )
                } catch (e: Exception) {
                    emitEvent(
                        "reply",
                        JSONObject()
                            .put("convId", convId)
                            .put("ok", false)
                            .put("error", e.message ?: "Request failed"),
                    )
                }
            }
        }
    }
}

package com.socraticchat.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

object StateStore {
    private fun stateFile(context: Context) = File(context.filesDir, "state.json")

    fun loadJson(context: Context): JSONObject {
        val file = stateFile(context)
        if (file.isFile) {
            try {
                val obj = JSONObject(file.readText(Charsets.UTF_8))
                if (obj.optJSONArray("conversations")?.length() ?: 0 > 0) {
                    return obj
                }
            } catch (_: Exception) {
            }
        }
        return defaultState(context)
    }

    fun saveJson(context: Context, state: JSONObject) {
        // Strip teacherKey before persisting so API keys are never written to disk.
        val copy = JSONObject(state.toString())
        copy.optJSONObject("config")?.remove("teacherKey")

        val target = stateFile(context)
        val tmp = File(context.filesDir, "state.json.tmp")
        tmp.writeText(copy.toString(2), Charsets.UTF_8)
        if (!tmp.renameTo(target)) {
            target.writeText(copy.toString(2), Charsets.UTF_8)
            tmp.delete()
        }
    }

    private fun defaultState(context: Context): JSONObject {
        val defaultModel = LlamaEngine.findModel(context, "")?.absolutePath ?: ""
        return JSONObject().apply {
            put(
                "config",
                JSONObject().apply {
                    put("ggufPath", defaultModel)
                    put("useTeacher", false)
                    put("teacherUrl", "")
                    put("teacherModel", "")
                    put("darkMode", true)
                },
            )
            put(
                "conversations",
                JSONArray().put(
                    JSONObject().apply {
                        put("id", 1)
                        put("title", "New Chat")
                        put("messages", JSONArray())
                    },
                ),
            )
            put("activeId", 1)
            put("nextId", 2)
        }
    }
}

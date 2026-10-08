<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { invoke, isTauri } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { getCurrentWebview } from '@tauri-apps/api/webview'
import { open, save } from '@tauri-apps/plugin-dialog'

type RuntimeStatus = {
  ready: boolean
  enginePath?: string | null
  modelCache?: string | null
  modelAvailable?: boolean
  message?: string
}

type ConversionEvent = {
  state: 'started' | 'starting' | 'loading' | 'processing' | 'encoding' | 'completed' | 'failed' | 'error' | 'cancelled'
  percent?: number | null
  processed?: number | null
  total?: number | null
  output?: string | null
  message?: string | null
  diagnostic?: string | null
}

const inputPath = ref('')
const outputPath = ref('')
const durationLimit = ref('')
const runtime = ref<RuntimeStatus | null>(null)
const task = ref<ConversionEvent | null>(null)
const busy = ref(false)
const errorMessage = ref('')
const dragging = ref(false)
const logLines = ref<string[]>([])
const unlisteners: UnlistenFn[] = []
const terminalStates = ['completed', 'failed', 'error', 'cancelled']

const inputName = computed(() => inputPath.value.split(/[\\/]/).pop() || '')
const progress = computed(() => {
  if (task.value?.state === 'completed') return 100
  const value = task.value?.percent
  if ((task.value?.total ?? 0) > 100000000) return 0
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0
})
const indeterminate = computed(() => busy.value && (task.value?.state === 'encoding' || typeof task.value?.percent !== 'number' || (task.value?.total ?? 0) > 100000000))
const taskTitle = computed(() => {
  switch (task.value?.state) {
    case 'started':
    case 'starting': return '准备转换'
    case 'loading': return '正在加载本地模型'
    case 'processing': return '正在生成深度视频'
    case 'encoding': return '正在编码并保存视频'
    case 'completed': return '转换完成'
    case 'failed':
    case 'error': return '转换失败'
    case 'cancelled': return '已取消'
    default: return '等待处理'
  }
})
const statusLabel = computed(() => {
  if (task.value?.state === 'completed') return '已完成'
  if (task.value?.state === 'failed' || task.value?.state === 'error') return '失败'
  if (task.value?.state === 'cancelled') return '已取消'
  return busy.value ? '处理中' : '待开始'
})
const progressLabel = computed(() => {
  if (task.value?.state === 'completed') return '100%'
  if (task.value?.state === 'encoding') return '正在保存'
  if (indeterminate.value) return '处理中…'
  return `${Math.round(progress.value)}%`
})
const runtimeLabel = computed(() => runtime.value === null ? '检查中' : runtime.value.ready === true ? '就绪' : '未就绪')
const canStart = computed(() => Boolean(inputPath.value.trim() && outputPath.value.trim() && !busy.value && runtime.value?.ready === true))

function addLog(message: string) {
  if (!message || logLines.value.at(-1) === message) return
  logLines.value = [...logLines.value.slice(-99), message]
}

function buildOutput(input: string) {
  const separator = input.includes('\\') ? '\\' : '/'
  const folder = input.replace(/[\\/][^\\/]*$/, '') || '.'
  const stem = (input.split(/[\\/]/).pop() || 'video').replace(/\.[^.]+$/, '')
  return `${folder.replace(/[\\/]$/, '')}${separator}${stem}_depth.mp4`
}

function selectInput(path: string) {
  if (busy.value) return
  if (!/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(path)) {
    errorMessage.value = '请选择 MP4、MOV、MKV、WebM、AVI 或 M4V 视频。'
    return
  }
  inputPath.value = path
  outputPath.value = buildOutput(path)
  task.value = null
  logLines.value = []
  errorMessage.value = ''
}

async function chooseInput() {
  if (busy.value) return
  errorMessage.value = ''
  if (!isTauri()) {
    errorMessage.value = '请通过桌面安装包运行此工具。'
    return
  }
  try {
    const selected = await open({
      multiple: false,
      title: '选择要转换的视频',
      filters: [{ name: '视频文件', extensions: ['mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v'] }],
    })
    if (typeof selected === 'string') selectInput(selected)
  } catch (error) {
    errorMessage.value = `无法选择视频：${String(error)}`
  }
}

async function chooseOutput() {
  if (busy.value || !isTauri()) return
  errorMessage.value = ''
  try {
    const selected = await save({
      title: '选择深度视频保存位置',
      defaultPath: outputPath.value || undefined,
      filters: [{ name: 'MP4 视频', extensions: ['mp4'] }],
    })
    if (typeof selected === 'string' && !busy.value) {
      outputPath.value = selected.toLowerCase().endsWith('.mp4') ? selected : `${selected}.mp4`
    }
  } catch (error) {
    errorMessage.value = `无法选择保存位置：${String(error)}`
  }
}

function updateTask(incoming: ConversionEvent, finished = false) {
  if (incoming.message) addLog(incoming.message)
  const previous = task.value
  if (previous && terminalStates.includes(previous.state) && !terminalStates.includes(incoming.state)) return
  task.value = {
    ...previous,
    ...incoming,
    processed: incoming.processed ?? previous?.processed,
    total: incoming.total ?? previous?.total,
    percent: incoming.state === 'completed' ? 100 : incoming.percent ?? previous?.percent,
    output: incoming.output ?? previous?.output,
  }
  if (finished || terminalStates.includes(incoming.state)) busy.value = false
  if (incoming.state === 'completed' && incoming.output) outputPath.value = incoming.output
  if (incoming.state === 'failed' || incoming.state === 'error') {
    errorMessage.value = incoming.message || '转换失败，请展开技术详情查看原因。'
  }
}

async function startConversion() {
  if (!canStart.value) return
  errorMessage.value = ''
  const seconds = durationLimit.value.trim() ? Number(durationLimit.value) : undefined
  if (seconds !== undefined && (!Number.isFinite(seconds) || seconds <= 0)) {
    errorMessage.value = '测试时长必须大于 0 秒，留空则转换完整视频。'
    return
  }
  busy.value = true
  dragging.value = false
  task.value = { state: 'starting', message: '正在启动本地转换引擎' }
  logLines.value = []
  try {
    await invoke<{ taskId: string; output: string }>('start_conversion', {
      request: { input: inputPath.value, output: outputPath.value, seconds, colormap: 'gray', device: 'auto' },
    })
  } catch (error) {
    updateTask({ state: 'failed', message: String(error) }, true)
  }
}

async function cancelConversion() {
  if (!busy.value || !isTauri()) return
  try {
    await invoke('cancel_conversion')
    addLog('已发送取消请求，正在清理当前进程…')
  } catch (error) {
    errorMessage.value = String(error)
  }
}

async function openPath(path: string) {
  if (!path || !isTauri()) return
  try {
    await invoke('open_path', { path })
  } catch (error) {
    errorMessage.value = String(error)
  }
}

async function loadRuntime() {
  if (!isTauri()) {
    runtime.value = { ready: false, message: '请通过桌面安装包启动应用。' }
    return
  }
  try {
    runtime.value = await invoke<RuntimeStatus>('check_runtime')
  } catch (error) {
    runtime.value = { ready: false, message: String(error) }
  }
}

onMounted(async () => {
  if (isTauri()) {
    try {
      unlisteners.push(await listen<ConversionEvent>('conversion://progress', (event) => updateTask(event.payload)))
      unlisteners.push(await listen<ConversionEvent>('conversion://finished', (event) => updateTask(event.payload, true)))
      unlisteners.push(await getCurrentWebview().onDragDropEvent((event) => {
        if (busy.value) {
          dragging.value = false
          return
        }
        const payload = event.payload
        dragging.value = payload.type === 'enter' || payload.type === 'over'
        if (payload.type === 'drop' && payload.paths[0]) selectInput(payload.paths[0])
      }))
    } catch (error) {
      runtime.value = { ready: false, message: '初始化桌面交互失败，请重新启动应用。' }
      errorMessage.value = String(error)
      return
    }
  }
  await loadRuntime()
})

onUnmounted(() => unlisteners.forEach((unlisten) => unlisten()))
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand-lockup"><div class="brand-mark" aria-hidden="true">D</div><div><h1>深度视频转换</h1><p class="brand-description">选择视频，一键生成灰度深度视频</p></div></div>
      <div class="local-badge"><span class="status-dot"></span>本机离线处理</div>
    </header>

    <section class="workspace-card">
      <div class="section-heading"><h2>选择视频</h2><span class="step-chip">输出 MP4 灰度视频</span></div>
      <div class="drop-zone" :class="{ dragging, disabled: busy }" :aria-disabled="busy">
        <div class="drop-icon" aria-hidden="true">↓</div>
        <div class="drop-copy"><strong>{{ inputName || '拖入视频，或点击右侧选择' }}</strong><span :title="inputPath">{{ inputPath || '支持 MP4、MOV、MKV、WebM、AVI、M4V' }}</span></div>
        <button class="primary-button" :disabled="busy" @click="chooseInput">选择视频</button>
      </div>

      <div class="options-grid">
        <div><label class="field-label" for="output">保存位置</label><div class="path-row"><input id="output" v-model="outputPath" type="text" :disabled="busy" placeholder="选择视频后自动生成" /><button class="secondary-button" :disabled="busy" @click="chooseOutput">更改</button></div><p class="field-hint">默认保存到原视频目录，文件名添加 <code>_depth</code>。</p></div>
        <div><label class="field-label" for="duration">只转换前几秒（可选）</label><input id="duration" v-model="durationLimit" type="number" min="1" step="1" :disabled="busy" placeholder="留空处理完整视频" /><p class="field-hint">可先填写 5 秒，查看效果和速度。</p></div>
      </div>

      <div class="action-row"><span class="action-hint">无需 API · 无需配置模型</span><div class="button-group"><button v-if="busy" class="secondary-button danger" @click="cancelConversion">取消转换</button><button class="convert-button" :disabled="!canStart" @click="startConversion">{{ busy ? '转换中…' : '开始转换' }} <span aria-hidden="true">→</span></button></div></div>
    </section>

    <div v-if="errorMessage" class="error-box" role="alert">{{ errorMessage }}<button v-if="task?.diagnostic && (task.state === 'failed' || task.state === 'error')" class="text-button" @click="openPath(task.diagnostic)">打开诊断日志</button></div>

    <section class="lower-grid">
      <div class="task-card" aria-live="polite">
        <div class="section-heading"><h2>{{ taskTitle }}</h2><span class="task-state" :class="task?.state">{{ statusLabel }}</span></div>
        <div class="progress-track" role="progressbar" aria-label="视频转换进度" aria-valuemin="0" aria-valuemax="100" :aria-valuenow="indeterminate ? undefined : progress"><span :class="{ indeterminate }" :style="{ width: `${progress}%` }"></span></div>
        <div class="task-meta"><span>{{ task?.processed != null ? `已处理 ${task.processed} 帧` : (busy ? '正在准备，请稍候' : '选择视频后即可开始') }}</span><span>{{ progressLabel }}</span></div>
        <div v-if="task?.output && task.state === 'completed'" class="completed-row"><span class="success-mark">✓</span><span class="filepath" :title="task.output">{{ task.output }}</span><button class="text-button" @click="openPath(task.output)">打开文件夹</button></div>
        <details v-if="logLines.length" class="technical-details"><summary>技术详情（运行日志）</summary><div class="log-lines"><pre v-for="(line, index) in logLines" :key="index">{{ line }}</pre></div></details>
      </div>
      <div class="runtime-card">
        <div class="section-heading"><h2>本地环境</h2><span :class="['runtime-state', { ready: runtime?.ready === true }]">{{ runtimeLabel }}</span></div>
        <p class="model-name">Depth Anything V2 Small</p>
        <p class="runtime-hint">{{ runtime?.message || '正在检查内置引擎和模型…' }}</p>
        <p class="offline-note">视频在本机处理，无需上传。</p>
      </div>
    </section>
  </main>
</template>

<template>
  <div class="app-shell">
    <div class="backdrop backdrop-a"></div>
    <div class="backdrop backdrop-b"></div>

    <main class="page">
      <AppHeader />

      <section class="stats-grid">
        <article class="stat-card">
          <span>页面状态</span>
          <strong>最终展示版</strong>
        </article>
        <article class="stat-card">
          <span>联调状态</span>
          <strong>{{ backendStatusText }}</strong>
        </article>
        <article class="stat-card">
          <span>最近结果</span>
          <strong>{{ latestLabel }}</strong>
        </article>
      </section>

      <section class="status-panel panel">
        <div class="panel-head">
          <div>
            <p class="panel-tag">Service Status</p>
            <h2>后端服务状态</h2>
          </div>
          <button class="ghost-button" type="button" @click="refreshBackendStatus">
            重新检测服务
          </button>
        </div>

        <div class="service-grid">
          <div class="result-metric">
            <span>接口可用性</span>
            <strong>{{ backendAvailable ? '在线' : '离线' }}</strong>
          </div>
          <div class="result-metric">
            <span>当前模式</span>
            <strong>{{ backendMode }}</strong>
          </div>
          <div class="result-metric">
            <span>样本数量</span>
            <strong>{{ backendSampleCount }}</strong>
          </div>
          <div class="result-metric">
            <span>F1 指标</span>
            <strong>{{ backendF1 }}</strong>
          </div>
        </div>
      </section>

      <section class="workspace-grid">
        <InputPanel
          v-model="inputText"
          :loading="loading"
          @submit="handleSubmit"
          @fill-example="fillExample"
        />
        <ResultPanel :result="result" :result-source="resultSource" />
      </section>

      <HistoryPanel :items="historyList" />
      <ModelSummaryPanel />
      <ScenarioPanel />
      <InfoPanel />
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import AppHeader from './components/AppHeader.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import InfoPanel from './components/InfoPanel.vue'
import InputPanel from './components/InputPanel.vue'
import ModelSummaryPanel from './components/ModelSummaryPanel.vue'
import ResultPanel from './components/ResultPanel.vue'
import ScenarioPanel from './components/ScenarioPanel.vue'
import { checkBackendHealth, predictText } from './api/predict'

const examples = [
  '这个群体根本不配得到尊重。',
  '我反对任何形式的歧视言论。',
  '不是我说，某些人真的是典中典。'
]

const inputText = ref('')
const loading = ref(false)
const result = ref(null)
const resultSource = ref('mock')
const historyList = ref([])
const backendAvailable = ref(false)
const backendHealth = ref(null)

const latestLabel = computed(() => {
  return result.value ? result.value.label_name : '暂无结果'
})

const backendStatusText = computed(() => {
  return backendAvailable.value ? '已连接后端' : '使用前端 mock'
})

const backendMode = computed(() => {
  if (!backendAvailable.value || !backendHealth.value) {
    return '未连接'
  }

  if (backendHealth.value.mode === 'artifact') {
    return '已加载训练产物'
  }

  if (backendHealth.value.mode === 'ml') {
    return '机器学习模式'
  }

  return '规则模式'
})

const backendSampleCount = computed(() => {
  if (!backendAvailable.value || !backendHealth.value) {
    return '--'
  }

  return `${backendHealth.value.sample_count}`
})

const backendF1 = computed(() => {
  if (!backendAvailable.value || !backendHealth.value?.metrics?.f1) {
    return '--'
  }

  return `${Math.round(backendHealth.value.metrics.f1 * 100)}%`
})

function fillExample() {
  const randomIndex = Math.floor(Math.random() * examples.length)
  inputText.value = examples[randomIndex]
}

async function refreshBackendStatus() {
  const healthResult = await checkBackendHealth()
  backendAvailable.value = healthResult.available
  backendHealth.value = healthResult.payload
}

async function handleSubmit() {
  const text = inputText.value.trim()

  if (!text || loading.value) {
    return
  }

  loading.value = true

  try {
    const response = await predictText(text)
    result.value = response
    resultSource.value = response.source

    historyList.value = [
      {
        id: `${Date.now()}`,
        text,
        label_name: response.label_name,
        scoreText: `${Math.round(response.score * 100)}%`
      },
      ...historyList.value
    ].slice(0, 5)

    backendAvailable.value = response.source === 'api'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  refreshBackendStatus()
})
</script>

<template>
  <div class="app-shell">
    <div class="backdrop backdrop-a"></div>
    <div class="backdrop backdrop-b"></div>

    <main class="page">
      <AppHeader />

      <section class="stats-grid">
        <article class="stat-card">
          <span>页面状态</span>
          <strong>可演示</strong>
        </article>
        <article class="stat-card">
          <span>联调状态</span>
          <strong>{{ resultSource === 'api' ? '已连接后端' : '等待后端' }}</strong>
        </article>
        <article class="stat-card">
          <span>最近结果</span>
          <strong>{{ latestLabel }}</strong>
        </article>
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
      <InfoPanel />
    </main>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import AppHeader from './components/AppHeader.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import InfoPanel from './components/InfoPanel.vue'
import InputPanel from './components/InputPanel.vue'
import ResultPanel from './components/ResultPanel.vue'
import { predictText } from './api/predict'

const examples = [
  '这个群体天生就比别人差，根本不配得到尊重。',
  '这家店的服务一般，但整体环境还是比较干净的。',
  '这种言论带有明显侮辱性，应该进行进一步审核。'
]

const inputText = ref('')
const loading = ref(false)
const result = ref(null)
const resultSource = ref('mock')
const historyList = ref([])

const latestLabel = computed(() => {
  return result.value ? result.value.label_name : '暂无结果'
})

function fillExample() {
  const randomIndex = Math.floor(Math.random() * examples.length)
  inputText.value = examples[randomIndex]
}

async function handleSubmit() {
  const text = inputText.value.trim()

  if (!text || loading.value) {
    return
  }

  loading.value = true

  try {
    const response = await predictText(text)
    const source = response?.message?.includes('前端演示') ? 'mock' : 'api'

    result.value = response
    resultSource.value = source

    historyList.value = [
      {
        id: `${Date.now()}`,
        text,
        label_name: response.label_name,
        scoreText: `${Math.round(response.score * 100)}%`
      },
      ...historyList.value
    ].slice(0, 5)
  } finally {
    loading.value = false
  }
}
</script>

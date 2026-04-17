<template>
  <section class="panel result-panel">
    <div class="panel-head">
      <div>
        <p class="panel-tag">Step 02</p>
        <h2>检测结果</h2>
      </div>
      <span class="helper-text">{{ sourceLabel }}</span>
    </div>

    <template v-if="result">
      <div class="result-hero" :class="result.label === 1 ? 'risk' : 'safe'">
        <p class="result-caption">分类标签</p>
        <h3>{{ result.label_name }}</h3>
        <p class="result-score">置信度：{{ confidenceText }}</p>
      </div>

      <div class="result-grid">
        <div class="result-metric">
          <span>标签值</span>
          <strong>{{ result.label }}</strong>
        </div>
        <div class="result-metric">
          <span>风险等级</span>
          <strong>{{ riskLevel }}</strong>
        </div>
      </div>

      <p class="result-message">{{ result.message }}</p>
    </template>

    <div v-else class="empty-state">
      <p>还没有检测结果。</p>
      <p>完成一次提交后，这里会展示分类标签、置信度和结果说明。</p>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: {
    type: Object,
    default: null
  },
  resultSource: {
    type: String,
    default: 'mock'
  }
})

const confidenceText = computed(() => {
  if (!props.result) {
    return '--'
  }

  return `${Math.round(props.result.score * 100)}%`
})

const riskLevel = computed(() => {
  if (!props.result) {
    return '--'
  }

  if (props.result.score >= 0.8) {
    return '高'
  }

  if (props.result.score >= 0.5) {
    return '中'
  }

  return '低'
})

const sourceLabel = computed(() => {
  return props.resultSource === 'api' ? '后端接口结果' : '前端演示结果'
})
</script>

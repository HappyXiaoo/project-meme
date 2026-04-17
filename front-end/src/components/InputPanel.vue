<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="panel-tag">Step 01</p>
        <h2>文本输入</h2>
      </div>
      <span class="helper-text">支持一句话或一整段文本</span>
    </div>

    <label class="field-label" for="detector-text">待检测文本</label>
    <textarea
      id="detector-text"
      :value="modelValue"
      class="text-input"
      placeholder="请输入待检测的中文文本，例如评论、短帖、弹幕或发言内容。"
      @input="$emit('update:modelValue', $event.target.value)"
    />

    <div class="input-footer">
      <p class="count-text">当前字数：{{ modelValue.length }}</p>
      <div class="action-row">
        <button class="ghost-button" type="button" @click="$emit('fill-example')">
          填充示例
        </button>
        <button
          class="primary-button"
          type="button"
          :disabled="loading || !modelValue.trim()"
          @click="$emit('submit')"
        >
          {{ loading ? '检测中...' : '开始检测' }}
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  modelValue: {
    type: String,
    default: ''
  },
  loading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['update:modelValue', 'submit', 'fill-example'])
</script>

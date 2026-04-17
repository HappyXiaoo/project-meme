<template>
  <section class="panel history-panel">
    <div class="panel-head">
      <div>
        <p class="panel-tag">Step 03</p>
        <h2>检测历史</h2>
      </div>
      <span class="helper-text">保留最近 5 条记录</span>
    </div>

    <div v-if="items.length" class="history-list">
      <article v-for="item in items" :key="item.id" class="history-item">
        <div class="history-top">
          <strong>{{ item.label_name }}</strong>
          <span>{{ item.scoreText }}</span>
        </div>
        <p class="history-text">{{ item.text }}</p>
      </article>
    </div>

    <div v-else class="empty-state">
      <p>历史记录区还是空的。</p>
      <p>后面做论文测试时，这里可以展示多次输入的系统输出结果。</p>
    </div>
  </section>
</template>

<script setup>
defineProps({
  items: {
    type: Array,
    default: () => []
  }
})
</script>

<template>
  <section class="panel summary-panel">
    <div class="panel-head">
      <div>
        <p class="panel-tag">Experiment Summary</p>
        <h2>模型结果概览</h2>
      </div>
      <span class="helper-text">基于正式测试集与人工测试集</span>
    </div>

    <div class="model-grid">
      <article class="model-card">
        <span>Baseline</span>
        <h3>TF-IDF + Logistic Regression</h3>
        <p>正式测试集 F1：78.87%</p>
        <p>特点：稳定、易解释，适合作为传统机器学习对照组。</p>
      </article>

      <article class="model-card">
        <span>Meme Features</span>
        <h3>模因式表达特征增强模型</h3>
        <p>正式测试集 F1：78.20%</p>
        <p>特点：总体指标略低，但在模因式攻击样例上更贴近课题目标。</p>
      </article>

      <article class="model-card highlight">
        <span>BERT</span>
        <h3>bert-base-chinese</h3>
        <p>正式测试集 F1：86.80%</p>
        <p>特点：正式测试集最优，但在人工模因测试集上不一定最强。</p>
      </article>
    </div>

    <div class="conclusion-box">
      <p>
        论文最终结论表明：BERT 更适合作为总体语义分类强基线，而模因特征模型更能体现“文本模因识别”这一课题的目标场景。
      </p>
    </div>
  </section>
</template>

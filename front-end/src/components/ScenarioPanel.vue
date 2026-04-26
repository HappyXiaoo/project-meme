<template>
  <section class="panel scenario-panel">
    <div class="panel-head">
      <div>
        <p class="panel-tag">Scenario Analysis</p>
        <h2>场景化结论</h2>
      </div>
      <span class="helper-text">人工测试集观察结果</span>
    </div>

    <div class="scenario-list">
      <article class="scenario-item">
        <strong>反对语境与描述语境</strong>
        <p>
          例如“我反对任何形式的歧视言论”“这种说法像是在侮辱别人”，当前系统已经能够较好地区分非直接攻击表达。
        </p>
      </article>

      <article class="scenario-item">
        <strong>模板化模因攻击表达</strong>
        <p>
          例如“不是我说，某些人真的是典中典”，模因特征模型对这类文本更敏感，更符合课题的场景定位。
        </p>
      </article>

      <article class="scenario-item">
        <strong>隐含群体攻击表达</strong>
        <p>
          例如“果然某些人又开始这一套了，笑死”，说明总体高指标模型并不一定最擅长目标场景识别。
        </p>
      </article>
    </div>
  </section>
</template>

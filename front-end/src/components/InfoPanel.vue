<template>
  <section class="info-grid">
    <article class="info-card">
      <p class="panel-tag">论文可写</p>
      <h3>界面与交互设计</h3>
      <p>
        这一版前端已经包含首页结构、输入流程、结果反馈和历史记录，后续可以直接截图写进系统设计章节。
      </p>
    </article>

    <article class="info-card">
      <p class="panel-tag">当前接口策略</p>
      <h3>优先保证流程跑通</h3>
      <p>
        页面优先尝试调用后端接口；如果后端未启动，会自动使用前端 mock 结果，确保你现在就能演示流程。
      </p>
    </article>

    <article class="info-card">
      <p class="panel-tag">后续扩展</p>
      <h3>可以平滑接入模型</h3>
      <p>
        等后端完成 `POST /predict` 后，前端基本不用大改，只需要对接返回字段即可进入联调阶段。
      </p>
    </article>
  </section>
</template>

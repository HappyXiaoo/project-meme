const API_BASE_URL = 'http://127.0.0.1:8000'

function buildMockResult(text) {
  const keywords = ['歧视', '仇恨', '滚', '低等', '恶心', '侮辱', '垃圾']
  const hitCount = keywords.reduce((count, keyword) => {
    return count + (text.includes(keyword) ? 1 : 0)
  }, 0)

  const isRisky = hitCount > 0 || text.length > 80
  const score = isRisky ? Math.min(0.62 + hitCount * 0.08, 0.96) : 0.18

  return {
    label: isRisky ? 1 : 0,
    label_name: isRisky ? '疑似有害文本' : '正常文本',
    score,
    message: isRisky
      ? '当前为前端演示判断结果，后续会由后端模型返回正式预测。'
      : '当前文本未触发演示规则，可作为正常样本展示。'
  }
}

export async function predictText(text) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    })

    if (!response.ok) {
      throw new Error('request_failed')
    }

    return await response.json()
  } catch (_error) {
    await new Promise((resolve) => setTimeout(resolve, 400))
    return buildMockResult(text)
  }
}

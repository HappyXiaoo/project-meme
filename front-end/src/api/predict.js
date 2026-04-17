const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8000'

const MOCK_KEYWORDS = [
  '歧视',
  '仇恨',
  '滚',
  '低等',
  '恶心',
  '侮辱',
  '垃圾',
  '不配'
]

function withTimeout(url, options = {}, timeoutMs = 4000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  return fetch(url, {
    ...options,
    signal: controller.signal
  }).finally(() => clearTimeout(timer))
}

function buildMockResult(text) {
  const hitCount = MOCK_KEYWORDS.reduce((count, keyword) => {
    return count + (text.includes(keyword) ? 1 : 0)
  }, 0)

  const isRisky = hitCount > 0 || text.length > 80
  const score = isRisky ? Math.min(0.62 + hitCount * 0.08, 0.96) : 0.18

  return {
    label: isRisky ? 1 : 0,
    label_name: isRisky ? '疑似有害文本' : '正常文本',
    score,
    message: isRisky
      ? '当前为前端演示结果，后续会由后端模型返回正式预测。'
      : '当前文本未触发演示规则，可作为正常样本展示。',
    source: 'mock'
  }
}

function normalizeApiResult(payload) {
  return {
    label: payload.label,
    label_name: payload.label_name,
    score: payload.score,
    message: payload.message,
    source: 'api'
  }
}

export async function checkBackendHealth() {
  try {
    const response = await withTimeout(`${API_BASE_URL}/health`, {
      method: 'GET'
    })

    if (!response.ok) {
      throw new Error('health_request_failed')
    }

    const payload = await response.json()
    return {
      available: true,
      payload
    }
  } catch (_error) {
    return {
      available: false,
      payload: null
    }
  }
}

export async function predictText(text) {
  try {
    const response = await withTimeout(
      `${API_BASE_URL}/predict`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      },
      5000
    )

    if (!response.ok) {
      throw new Error('predict_request_failed')
    }

    const payload = await response.json()
    return normalizeApiResult(payload)
  } catch (_error) {
    await new Promise((resolve) => setTimeout(resolve, 300))
    return buildMockResult(text)
  }
}

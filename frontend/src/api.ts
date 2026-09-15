import type { Game, StreamEvent } from './types'

export interface ChatContext {
  candidates: { appid: number; name: string }[]
}

export async function importLibrary(mode: string, value: string): Promise<Game[]> {
  const r = await fetch('/api/library', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, value }),
  })
  if (!r.ok) {
    const detail = await r.json().catch(() => null)
    throw new Error(detail?.detail || '读取游戏库失败')
  }
  const data = await r.json()
  return data.games as Game[]
}

export async function streamChat(
  message: string,
  library: Game[],
  context: ChatContext,
  onEvent: (ev: StreamEvent) => void,
): Promise<void> {
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, library, context }),
  })
  if (!r.ok || !r.body) {
    const detail = await r.json().catch(() => null)
    throw new Error(detail?.detail || '请求失败')
  }

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() || ''
    for (const chunk of chunks) {
      const line = chunk.trim()
      if (!line.startsWith('data: ')) continue
      try {
        const ev = JSON.parse(line.slice(6)) as StreamEvent
        onEvent(ev)
      } catch {
        // 忽略无法解析的行
      }
    }
  }
}

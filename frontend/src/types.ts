export interface Game {
  appid: number
  name: string
  playtime_forever: number | null
}

export interface DlcItem {
  appid: number
  name: string
  owned: boolean
  price_cents: number | null
  currency: string | null
}

export interface DlcRow {
  appid: number
  name: string
  owned_dlc: DlcItem[]
  missing_dlc: DlcItem[]
  missing_count: number
}

export interface DlcReport {
  games_with_dlc: number
  total_missing_dlc: number
  total_missing_price_cents: number
  rows: DlcRow[]
}

export interface Candidate {
  appid: number
  name: string
  genres: string
  categories: string
  price_cents: number
  positive: number
  metacritic_score: number
  distance: number
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Step {
  node: string
  status: 'start' | 'end'
}

export type StreamEvent =
  | { type: 'node'; data: { node: string; status: 'start' | 'end' } }
  | { type: 'token'; node: string; data: string }
  | {
      type: 'done'
      data: {
        text: string
        intent: string | null
        dlc_report: DlcReport | null
        candidates: Candidate[] | null
      }
    }
  | { type: 'error'; data: string }

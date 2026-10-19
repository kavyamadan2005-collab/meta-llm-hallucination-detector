const DEFAULT_API_BASE_URL =
  typeof window === 'undefined'
    ? 'http://127.0.0.1:8000'
    : `http://${window.location.hostname}:8000`

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') || DEFAULT_API_BASE_URL

export type Verdict = 'SUPPORTED' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE' | 'UNSUPPORTED' | (string & {})
export type VerdictKey = 'supported' | 'contradicted' | 'insufficient'

export interface AnalyzeRequest {
  question: string
}

export interface Claim {
  claim_id: string
  claim_text: string
  original_sentence?: string
  sentence_index?: number
}

export interface Evidence {
  source_title?: string | null
  source_url?: string | null
  relevance_score?: number | null
  source?: string | null
  title?: string | null
  url?: string | null
  snippet?: string | null
  text?: string | null
  score?: number | null
  [key: string]: unknown
}

export interface VerificationResult {
  claim_id: string
  claim_text: string
  verdict: Verdict
  confidence: number
  explanation?: string | null
  evidence?: Evidence[] | null
  supporting_evidence?: Evidence[] | null
  contradicting_evidence?: Evidence[] | null
  event_fact?: VerificationAssessment | null
  attributes?: AttributeVerification[] | null
}

export interface VerificationAssessment {
  verdict: Verdict
  confidence: number
  explanation?: string | null
  supporting_evidence?: Evidence[] | null
  contradicting_evidence?: Evidence[] | null
}

export interface AttributeVerification extends VerificationAssessment {
  name: string
  value: string
}

export interface OverallResult {
  status: string
  confidence: number
  explanation: string
}

export interface LLMResponse {
  question?: string
  response_text: string
  model?: string | null
  timestamp?: string
}

export interface AnalyzeResponse {
  question: string
  llm_response: LLMResponse
  claims?: Claim[] | null
  graph?: Record<string, unknown> | null
  verification_results?: VerificationResult[] | null
  overall_result?: OverallResult | null
}

export class ApiError extends Error {
  status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      signal,
      cache: 'no-store',
    })
    return res.ok
  } catch {
    return false
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function readString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function readNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function toClaim(value: unknown): Claim | null {
  if (!isObject(value)) return null
  const claimId = readString(value.claim_id)
  const claimText = readString(value.claim_text)
  if (!claimId || !claimText) return null
  return {
    claim_id: claimId,
    claim_text: claimText,
    original_sentence: readString(value.original_sentence) || undefined,
    sentence_index: readNumber(value.sentence_index) ?? undefined,
  }
}

function toEvidence(value: unknown): Evidence | null {
  if (!isObject(value)) return null
  return {
    source_title: readString(value.source_title),
    source_url: readString(value.source_url),
    relevance_score: readNumber(value.relevance_score),
    source: readString(value.source),
    title: readString(value.title),
    url: readString(value.url),
    snippet: readString(value.snippet),
    text: readString(value.text),
    score: readNumber(value.score),
  }
}

function toVerificationAssessment(value: unknown): VerificationAssessment | null {
  if (!isObject(value)) return null
  const verdict = readString(value.verdict)
  if (!verdict) return null
  return {
    verdict,
    confidence: readNumber(value.confidence) ?? 0,
    explanation: readString(value.explanation),
    supporting_evidence: Array.isArray(value.supporting_evidence)
      ? value.supporting_evidence.map(toEvidence).filter((item): item is Evidence => item !== null)
      : [],
    contradicting_evidence: Array.isArray(value.contradicting_evidence)
      ? value.contradicting_evidence.map(toEvidence).filter((item): item is Evidence => item !== null)
      : [],
  }
}

function toVerificationResult(value: unknown): VerificationResult | null {
  if (!isObject(value)) return null
  const claimId = readString(value.claim_id)
  const claimText = readString(value.claim_text)
  const verdict = readString(value.verdict)
  if (!claimId || !claimText || !verdict) return null

  const evidence = Array.isArray(value.evidence)
    ? value.evidence.map(toEvidence).filter((item): item is Evidence => item !== null)
    : []
  const supportingEvidence = Array.isArray(value.supporting_evidence)
    ? value.supporting_evidence.map(toEvidence).filter((item): item is Evidence => item !== null)
    : []
  const contradictingEvidence = Array.isArray(value.contradicting_evidence)
    ? value.contradicting_evidence.map(toEvidence).filter((item): item is Evidence => item !== null)
    : []
  const eventFact = toVerificationAssessment(value.event_fact)
  const attributes = Array.isArray(value.attributes)
    ? value.attributes.flatMap((attribute): AttributeVerification[] => {
        if (!isObject(attribute)) return []
        const assessment = toVerificationAssessment(attribute)
        const name = readString(attribute.name)
        const attributeValue = readString(attribute.value)
        return assessment && name && attributeValue
          ? [{ ...assessment, name, value: attributeValue }]
          : []
      })
    : []

  return {
    claim_id: claimId,
    claim_text: claimText,
    verdict,
    confidence: readNumber(value.confidence) ?? 0,
    explanation: readString(value.explanation),
    evidence,
    supporting_evidence: supportingEvidence,
    contradicting_evidence: contradictingEvidence,
    event_fact: eventFact,
    attributes,
  }
}

function normalizeAnalyzeResponse(payload: unknown): AnalyzeResponse {
  if (!isObject(payload)) {
    throw new ApiError('Unexpected response from backend. Please try again.')
  }

  const question = readString(payload.question)
  const llmResponse = isObject(payload.llm_response) ? payload.llm_response : null
  const responseText = llmResponse ? readString(llmResponse.response_text) : null
  if (!question || !responseText) {
    throw new ApiError('Backend response is missing required fields.')
  }

  const claims = Array.isArray(payload.claims)
    ? payload.claims.map(toClaim).filter((item): item is Claim => item !== null)
    : []

  const verificationResults = Array.isArray(payload.verification_results)
    ? payload.verification_results
        .map(toVerificationResult)
        .filter((item): item is VerificationResult => item !== null)
    : []
  const overall = isObject(payload.overall_result)
    ? {
        status: readString(payload.overall_result.status) || 'INCONCLUSIVE',
        confidence: readNumber(payload.overall_result.confidence) ?? 0,
        explanation: readString(payload.overall_result.explanation) || '',
      }
    : null

  return {
    question,
    llm_response: {
      question: readString(llmResponse?.question) || undefined,
      response_text: responseText,
      model: readString(llmResponse?.model),
      timestamp: readString(llmResponse?.timestamp) || undefined,
    },
    claims,
    graph: isObject(payload.graph) ? payload.graph : null,
    verification_results: verificationResults,
    overall_result: overall,
  }
}

function explainStatusError(status: number, detail: string): string {
  if (status === 400) {
    return detail || 'Please enter a valid question before analyzing.'
  }
  if (status === 500) {
    return 'The backend hit an internal error. Please retry in a moment.'
  }
  if (status === 502) {
    return 'The language model provider failed while generating an answer.'
  }
  if (status === 503) {
    return 'The analysis pipeline is currently unavailable. Please try again shortly.'
  }
  return detail || `Backend returned ${status}.`
}

export async function analyzeQuestion(
  question: string,
  signal?: AbortSignal,
): Promise<AnalyzeResponse> {
  let res: Response
  try {
    const payload: AnalyzeRequest = { question }
    res = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
      cache: 'no-store',
    })
  } catch (error) {
    if ((error as Error)?.name === 'AbortError') throw error
    throw new ApiError(
      `Could not reach the analysis backend at ${API_BASE_URL}. Make sure the FastAPI server is running.`,
    )
  }

  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? '')
    } catch {
      detail = await res.text().catch(() => '')
    }
    throw new ApiError(explainStatusError(res.status, detail), res.status)
  }

  let payload: unknown
  try {
    payload = await res.json()
  } catch {
    throw new ApiError('Backend returned invalid JSON. Please try again.')
  }

  return normalizeAnalyzeResponse(payload)
}

export function verdictKey(verdict: Verdict): VerdictKey {
  const v = String(verdict || '').toUpperCase()
  if (v.includes('SUPPORT') && !v.includes('UN') && !v.includes('NOT')) return 'supported'
  if (v.includes('CONTRADICT') || v.includes('REFUT')) return 'contradicted'
  return 'insufficient'
}

export function summarize(results: VerificationResult[]) {
  const counts = { supported: 0, contradicted: 0, insufficient: 0 }
  for (const r of results) counts[verdictKey(r.verdict)] += 1
  const total = results.length
  const avgConfidence = total
    ? results.reduce((sum, r) => sum + (Number(r.confidence) || 0), 0) / total
    : 0
  return { counts, total, avgConfidence }
}

export function formatEvidenceText(evidence: Evidence): string {
  return (
    (typeof evidence.text === 'string' && evidence.text) ||
    (typeof evidence.snippet === 'string' && evidence.snippet) ||
    (typeof evidence.content === 'string' && (evidence.content as string)) ||
    ''
  )
}

export function formatEvidenceTitle(evidence: Evidence, index: number): string {
  return (
    (typeof evidence.source_title === 'string' && evidence.source_title) ||
    (typeof evidence.title === 'string' && evidence.title) ||
    (typeof evidence.source === 'string' && evidence.source) ||
    (typeof evidence.source_url === 'string' && safeHost(evidence.source_url)) ||
    (typeof evidence.url === 'string' && safeHost(evidence.url)) ||
    `Evidence ${index + 1}`
  )
}

export function safeHost(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

'use client'

import { useRef, useState } from 'react'
import { AlertTriangle, BookOpen, FlaskConical, RotateCcw, ShieldCheck } from 'lucide-react'
import { analyzeQuestion, type AnalyzeResponse } from '@/lib/api'
import { AnalysisProgress } from '@/components/analysis-progress'
import { BackendStatus } from '@/components/backend-status'
import { QuestionForm } from '@/components/question-form'
import { ResultsView } from '@/components/results-view'

export default function HomePage() {
  const [question, setQuestion] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const runAnalysis = async (value: string) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setSubmitted(value)
    setLoading(true)
    setError(null)
    setData(null)
    try {
      setData(await analyzeQuestion(value, controller.signal))
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return
      setError((err as Error).message || 'Analysis failed.')
    } finally {
      if (controllerRef.current === controller) setLoading(false)
    }
  }

  const reset = () => {
    controllerRef.current?.abort()
    setData(null)
    setError(null)
    setSubmitted('')
    setQuestion('')
    setLoading(false)
  }

  return (
    <div className="min-h-dvh">
      <header className="border-b border-border bg-card/90">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm"><ShieldCheck className="size-5" /></div>
            <div><p className="text-sm font-bold tracking-tight">Meta-LLM Hallucination Detection</p><p className="text-xs text-muted-foreground">Explainable claim-level verification for LLM responses</p></div>
          </div>
          <div className="flex items-center gap-3"><span className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex"><BookOpen className="size-3.5" /> Research dashboard</span><BackendStatus /></div>
        </div>
      </header>

      <main className="relative mx-auto max-w-7xl px-5 pb-16 pt-8 lg:px-8">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-72 opacity-40 [mask-image:linear-gradient(to_bottom,black,transparent)]"><div className="grid-backdrop h-full" /></div>
        <div className="relative">
          {!data && !loading && (
            <section className="grid items-center gap-10 py-8 lg:grid-cols-[1fr_0.78fr] lg:py-14">
              <div><div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1.5 text-xs font-semibold text-primary"><FlaskConical className="size-3.5" /> AI reliability workspace</div><h1 className="mt-5 max-w-2xl text-4xl font-bold leading-[1.08] tracking-tight text-balance sm:text-6xl">See what an AI answer is really made of.</h1><p className="mt-5 max-w-xl text-base leading-7 text-muted-foreground sm:text-lg">Ask a question and follow the complete path from generated response to extracted claims, retrieved evidence, and transparent verdicts.</p><div className="mt-7 flex flex-wrap gap-3 text-xs font-medium text-muted-foreground"><span className="rounded-lg border border-border bg-card px-3 py-2">Claim-level analysis</span><span className="rounded-lg border border-border bg-card px-3 py-2">Evidence provenance</span><span className="rounded-lg border border-border bg-card px-3 py-2">Local or web-backed</span></div></div>
              <section className="rounded-2xl border border-border bg-card p-5 shadow-lg shadow-primary/5 sm:p-6"><p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Start an analysis</p><h2 className="mt-2 text-xl font-semibold">Analyze an LLM response</h2><p className="mt-2 text-sm leading-relaxed text-muted-foreground">The backend will generate an answer and verify its factual claims.</p><div className="mt-5"><QuestionForm question={question} onQuestionChange={setQuestion} onSubmit={runAnalysis} loading={loading} /></div></section>
            </section>
          )}
          {loading && <div className="mx-auto max-w-5xl py-10"><AnalysisProgress question={submitted} /></div>}
          {error && !loading && <div role="alert" className="mx-auto mt-8 flex max-w-3xl items-start gap-3 rounded-2xl border border-contradicted/25 bg-contradicted/5 p-5 text-sm"><AlertTriangle className="mt-0.5 size-5 shrink-0 text-contradicted" /><div><p className="font-semibold">Analysis could not be completed</p><p className="mt-1 leading-relaxed text-muted-foreground">{error}</p><button type="button" onClick={() => runAnalysis(submitted)} className="mt-3 inline-flex items-center gap-2 font-semibold text-primary hover:underline"><RotateCcw className="size-3.5" /> Try again</button></div></div>}
          {data && !loading && <div className="pt-2"><ResultsView data={data} onReset={reset} /></div>}
        </div>
      </main>
      <footer className="border-t border-border bg-card"><div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-5 py-5 text-xs text-muted-foreground lg:px-8"><span>Meta · Explainable hallucination detection</span><span>FastAPI analysis pipeline</span></div></footer>
    </div>
  )
}

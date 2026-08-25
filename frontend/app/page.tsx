'use client'

import { useRef, useState } from 'react'
import { TriangleAlert } from 'lucide-react'
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
      const result = await analyzeQuestion(value, controller.signal)
      setData(result)
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
    <div className="relative min-h-dvh overflow-hidden">
      <div
        className="grid-backdrop pointer-events-none absolute inset-x-0 top-0 h-[520px] opacity-60 [mask-image:radial-gradient(70%_60%_at_50%_0%,black,transparent)]"
        aria-hidden="true"
      />

      <header className="relative z-10 mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-3">
          <span
            className="flex size-8 items-center justify-center rounded-md border border-primary/40 bg-primary/10 font-mono text-sm font-semibold text-primary"
            aria-hidden="true"
          >
            M
          </span>
          <div className="leading-tight">
            <p className="font-mono text-sm font-semibold tracking-[0.28em]">META</p>
            <p className="text-[11px] text-muted-foreground">Hallucination Detection</p>
          </div>
        </div>
        <BackendStatus />
      </header>

      <main className="relative z-10 mx-auto w-full max-w-5xl px-6 pb-24">
        {!data && !loading && (
          <section className="pt-10 sm:pt-20">
            <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-primary">
              Claim-level verification engine
            </p>
            <h1 className="mt-5 text-balance text-4xl font-medium leading-[1.05] tracking-tight sm:text-6xl">
              Verify AI answers,
              <br />
              <span className="text-muted-foreground">claim by claim.</span>
            </h1>
            <p className="mt-6 max-w-2xl text-pretty text-lg leading-relaxed text-muted-foreground">
              Meta generates an answer with your local model, decomposes it into atomic claims,
              retrieves evidence for each one, and returns a verdict with confidence and sources.
            </p>

            <div className="mt-10">
              <QuestionForm
                question={question}
                onQuestionChange={setQuestion}
                onSubmit={runAnalysis}
                loading={loading}
              />
            </div>
          </section>
        )}

        {loading && (
          <div className="pt-10">
            <AnalysisProgress question={submitted} />
          </div>
        )}

        {error && !loading && (
          <div
            role="alert"
            className="animate-rise mt-8 flex items-start gap-3 rounded-2xl border border-destructive/40 bg-destructive/10 p-5"
          >
            <TriangleAlert className="mt-0.5 size-5 shrink-0 text-destructive" aria-hidden="true" />
            <div className="min-w-0">
              <p className="text-sm font-medium">Analysis failed</p>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{error}</p>
            </div>
          </div>
        )}

        {data && !loading && (
          <div className="pt-10">
            <ResultsView data={data} onReset={reset} />
          </div>
        )}
      </main>

      <footer className="relative z-10 border-t border-border">
        <div className="mx-auto flex w-full max-w-5xl flex-wrap items-center justify-between gap-2 px-6 py-6 font-mono text-[11px] text-muted-foreground/70">
          <span>META · LLM Hallucination Detection System</span>
          <span>FastAPI · Local Analysis Pipeline</span>
        </div>
      </footer>
    </div>
  )
}

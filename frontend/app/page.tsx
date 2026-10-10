'use client'

import { useEffect, useRef, useState } from 'react'
import {
  Clock3,
  Menu,
  Moon,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from 'lucide-react'
import { analyzeQuestion, type AnalyzeResponse } from '@/lib/api'
import { AnalysisProgress } from '@/components/analysis-progress'
import { BackendStatus } from '@/components/backend-status'
import { QuestionForm } from '@/components/question-form'
import { ResultsView, type ResearchStage } from '@/components/results-view'

const HISTORY_KEY = 'meta-llm-analysis-history'

export default function HomePage() {
  const [question, setQuestion] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<string[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [processOpen, setProcessOpen] = useState(false)
  const [stage, setStage] = useState<ResearchStage>(0)
  const [darkMode, setDarkMode] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(HISTORY_KEY)
      if (stored) setHistory(JSON.parse(stored))
      const savedTheme = window.localStorage.getItem('meta-llm-theme')
      const shouldUseDark = savedTheme === 'dark'
      setDarkMode(shouldUseDark)
      document.documentElement.classList.toggle('dark', shouldUseDark)
    } catch {
      // Local preferences are optional.
    }
  }, [])

  const toggleTheme = () => {
    const next = !darkMode
    setDarkMode(next)
    document.documentElement.classList.toggle('dark', next)
    window.localStorage.setItem('meta-llm-theme', next ? 'dark' : 'light')
  }

  const rememberQuestion = (value: string) => {
    const next = [value, ...history.filter((item) => item !== value)].slice(0, 12)
    setHistory(next)
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(next))
  }

  const runAnalysis = async (value: string) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setSubmitted(value)
    setQuestion('')
    setLoading(true)
    setError(null)
    setData(null)
    setProcessOpen(false)
    setStage(0)
    rememberQuestion(value)
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
    setProcessOpen(false)
    setStage(0)
  }

  const selectHistory = (item: string) => {
    setSidebarOpen(false)
    runAnalysis(item)
  }

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <div className="flex min-h-dvh">
        {sidebarOpen && (
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 z-30 bg-slate-950/40 backdrop-blur-sm lg:hidden"
          />
        )}
        <aside className={`fixed inset-y-0 left-0 z-40 flex w-72 shrink-0 flex-col border-r border-border bg-card/95 p-4 backdrop-blur transition-transform lg:static lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
          <div className="flex items-center justify-between px-2 py-2">
            <div className="flex items-center gap-3">
              <span className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm"><ShieldCheck className="size-5" /></span>
              <div><p className="text-sm font-bold tracking-tight">Meta-LLM</p><p className="text-[11px] text-muted-foreground">Hallucination Detection</p></div>
            </div>
            <button type="button" onClick={() => setSidebarOpen(false)} className="rounded-lg p-2 text-muted-foreground hover:bg-muted lg:hidden" aria-label="Close sidebar"><X className="size-4" /></button>
          </div>
          <button type="button" onClick={reset} className="mt-7 flex h-10 items-center gap-2 rounded-xl border border-primary/25 bg-primary/10 px-3 text-sm font-semibold text-primary transition-colors hover:bg-primary/15"><Plus className="size-4" /> New analysis</button>
          <div className="mt-8 flex items-center justify-between px-2"><p className="font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground">History</p><Clock3 className="size-3.5 text-muted-foreground" /></div>
          <div className="mt-3 flex-1 space-y-1 overflow-y-auto">
            {history.length ? history.map((item) => (
              <button key={item} type="button" onClick={() => selectHistory(item)} className="w-full rounded-xl px-3 py-2.5 text-left text-sm leading-5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">{item}</button>
            )) : <p className="px-3 py-3 text-xs leading-5 text-muted-foreground/70">Your analyzed questions will appear here.</p>}
          </div>
          <div className="mt-4 border-t border-border pt-4">
            <button type="button" onClick={toggleTheme} className="flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground">
              <span className="flex items-center gap-2">{darkMode ? <Moon className="size-4" /> : <Sun className="size-4" />} {darkMode ? 'Dark mode' : 'Light mode'}</span>
              <span className="font-mono text-[10px] uppercase tracking-wider">Toggle</span>
            </button>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur-xl">
            <div className="flex h-16 items-center justify-between gap-4 px-4 sm:px-6 lg:px-10">
              <div className="flex items-center gap-3">
                <button type="button" onClick={() => setSidebarOpen(true)} className="rounded-lg p-2 text-muted-foreground hover:bg-muted lg:hidden" aria-label="Open sidebar"><Menu className="size-5" /></button>
                <div><p className="text-sm font-semibold">AI research workspace</p><p className="hidden text-xs text-muted-foreground sm:block">Trace answers from generation to verification</p></div>
              </div>
              <div className="flex items-center gap-2 sm:gap-4"><BackendStatus /><button type="button" onClick={toggleTheme} className="rounded-lg border border-border p-2 text-muted-foreground hover:bg-muted hover:text-foreground" aria-label="Toggle theme">{darkMode ? <Sun className="size-4" /> : <Moon className="size-4" />}</button></div>
            </div>
          </header>

          <div className="grid-backdrop min-h-[calc(100dvh-4rem)]">
            <div className="mx-auto flex w-full max-w-5xl flex-col px-4 pb-10 pt-8 sm:px-6 lg:px-10">
              {!submitted && !data && !loading && (
                <section className="flex flex-1 flex-col justify-center py-14 sm:py-24">
                  <div className="mx-auto max-w-3xl text-center">
                    <div className="mx-auto flex size-14 items-center justify-center rounded-2xl border border-primary/20 bg-primary/10 text-primary"><Sparkles className="size-6" /></div>
                    <p className="mt-6 font-mono text-[11px] font-bold uppercase tracking-[0.22em] text-primary">Evidence-backed answers</p>
                    <h1 className="mt-4 text-4xl font-bold tracking-tight text-balance sm:text-6xl">Ask anything. See what the answer is made of.</h1>
                    <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">Generate an answer, extract its factual claims, and inspect the evidence and verdicts in one transparent workspace.</p>
                  </div>
                  <div className="mx-auto mt-10 w-full max-w-3xl"><QuestionForm question={question} onQuestionChange={setQuestion} onSubmit={runAnalysis} loading={loading} /></div>
                </section>
              )}

              {(submitted || loading || data || error) && (
                <section className="space-y-6">
                  <div className="flex justify-end border-b border-border pb-6">
                    <div className="max-w-2xl rounded-2xl rounded-br-md bg-secondary px-5 py-4">
                      <p className="flex items-center justify-end gap-2 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground"><Search className="size-3.5" /> You asked</p>
                      <p className="mt-2 text-right text-lg leading-7">{submitted}</p>
                    </div>
                  </div>
                  {loading && (
                    <div className="space-y-4">
                      <button type="button" onClick={() => setProcessOpen((value) => !value)} className="flex w-full items-center justify-between rounded-2xl border border-primary/20 bg-primary/5 px-4 py-3 text-left text-sm text-primary transition-colors hover:bg-primary/10" aria-expanded={processOpen}>
                        <span className="flex items-center gap-3"><span className="flex gap-1" aria-hidden="true"><span className="size-1.5 animate-pulse rounded-full bg-primary" /><span className="size-1.5 animate-pulse rounded-full bg-primary [animation-delay:150ms]" /><span className="size-1.5 animate-pulse rounded-full bg-primary [animation-delay:300ms]" /></span><span><span className="block font-semibold">Generating response...</span><span className="mt-0.5 block text-xs text-primary/70">Research pipeline active</span></span></span>
                        <span className="text-xs font-semibold">{processOpen ? 'Hide process' : 'View process'}</span>
                      </button>
                      {processOpen && <AnalysisProgress question={submitted} />}
                    </div>
                  )}
                  {error && !loading && <div role="alert" className="rounded-2xl border border-contradicted/25 bg-contradicted/5 p-5 text-sm"><p className="font-semibold text-contradicted">Analysis could not be completed</p><p className="mt-1 text-muted-foreground">{error}</p><button type="button" onClick={() => runAnalysis(submitted)} className="mt-3 font-semibold text-primary hover:underline">Try again</button></div>}
                  {data && !loading && <ResultsView data={data} stage={stage} onStageChange={setStage} onReset={reset} />}
                  {!loading && !data && !error && <div className="pt-6"><QuestionForm question={question} onQuestionChange={setQuestion} onSubmit={runAnalysis} loading={loading} /></div>}
                </section>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

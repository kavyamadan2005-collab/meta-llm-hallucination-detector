'use client'

import { useEffect, useState } from 'react'
import { Check, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

const STAGES = [
  { label: 'Question', detail: 'Input received' },
  { label: 'LLM response', detail: 'Generating an answer' },
  { label: 'Claim extraction', detail: 'Finding factual statements' },
  { label: 'Evidence retrieval', detail: 'Searching configured sources' },
  { label: 'Claim verification', detail: 'Comparing claims with evidence' },
]

/**
 * The backend exposes a single blocking /analyze call, so this reflects the
 * pipeline stages over elapsed time. The final stage stays active until the
 * real response arrives.
 */
export function AnalysisProgress({ question }: { question: string }) {
  const [stage, setStage] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const timers = [
      setTimeout(() => setStage(1), 2500),
      setTimeout(() => setStage(2), 6000),
      setTimeout(() => setStage(3), 10000),
    ]
    const tick = setInterval(() => setElapsed((value) => value + 1), 1000)
    return () => {
      timers.forEach(clearTimeout)
      clearInterval(tick)
    }
  }, [])

  return (
    <section
      aria-live="polite"
      className="animate-rise rounded-2xl border border-border bg-card/60 p-6 backdrop-blur sm:p-8"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-2xl">
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">
            Analysis in progress
          </p>
          <h2 className="mt-2 text-pretty text-xl font-medium leading-snug">{question}</h2>
        </div>
        <p className="font-mono text-sm text-muted-foreground">
          {String(Math.floor(elapsed / 60)).padStart(2, '0')}:
          {String(elapsed % 60).padStart(2, '0')}
        </p>
      </div>

      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        The backend exposes one request, so the stepper communicates the analysis pipeline while the response is being prepared.
      </p>
      <div className="mt-8 grid gap-2 sm:grid-cols-5">
        {STAGES.map((item, index) => {
          const done = index < stage
          const active = index === stage
          return (
            <div
              key={item.label}
              className={cn(
                'relative rounded-xl border border-border px-3 py-4 transition-colors',
                active && 'bg-accent/40',
              )}
            >
              <div className="flex items-center gap-2">
                {done ? (
                  <span className="flex size-5 items-center justify-center rounded-full bg-supported/15">
                    <Check className="size-3 text-supported" aria-hidden="true" strokeWidth={3} />
                  </span>
                ) : active ? (
                  <Loader2 className="size-4 animate-spin text-primary" aria-hidden="true" />
                ) : (
                  <span className="size-1.5 rounded-full bg-muted-foreground/40" />
                )}
              </div>
              <div className="mt-3 min-w-0">
                <p
                  className={cn(
                    'text-sm font-medium transition-colors',
                    done && 'text-muted-foreground',
                    active && 'text-foreground',
                    !done && !active && 'text-muted-foreground/50',
                  )}
                >
                  {item.label}
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground/70">{item.detail}</p>
                {active && <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-primary/10"><div className="animate-sweep h-full w-1/3 bg-primary" /></div>}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

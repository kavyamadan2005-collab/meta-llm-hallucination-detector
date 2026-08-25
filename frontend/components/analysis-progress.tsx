'use client'

import { useEffect, useState } from 'react'
import { Check, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

const STAGES = [
  { label: 'Generating answer', detail: 'Querying the local language model' },
  { label: 'Extracting claims', detail: 'Decomposing the answer into atomic facts' },
  { label: 'Retrieving evidence', detail: 'Searching sources for each claim' },
  { label: 'Verifying claims', detail: 'Comparing claims against retrieved evidence' },
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
            Analyzing your question...
          </p>
          <h2 className="mt-2 text-pretty text-xl font-medium leading-snug">{question}</h2>
        </div>
        <p className="font-mono text-sm text-muted-foreground">
          {String(Math.floor(elapsed / 60)).padStart(2, '0')}:
          {String(elapsed % 60).padStart(2, '0')}
        </p>
      </div>

      <div className="mt-8 space-y-1">
        {STAGES.map((item, index) => {
          const done = index < stage
          const active = index === stage
          return (
            <div
              key={item.label}
              className={cn(
                'flex items-start gap-4 rounded-xl px-3 py-3 transition-colors',
                active && 'bg-accent/40',
              )}
            >
              <div className="mt-0.5 flex size-6 shrink-0 items-center justify-center">
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
              <div className="min-w-0 flex-1">
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
                {active && (
                  <div className="mt-3 h-px w-full overflow-hidden bg-border">
                    <div className="animate-sweep h-px w-1/3 bg-primary" />
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

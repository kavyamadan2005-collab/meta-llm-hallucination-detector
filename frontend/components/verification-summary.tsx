'use client'

import { summarize, type VerificationResult } from '@/lib/api'

export function VerificationSummary({ results }: { results: VerificationResult[] }) {
  const { counts, total, avgConfidence, trust } = summarize(results)

  const segments = [
    { key: 'supported', label: 'Supported', value: counts.supported, color: 'bg-supported' },
    {
      key: 'contradicted',
      label: 'Contradicted',
      value: counts.contradicted,
      color: 'bg-contradicted',
    },
    {
      key: 'unsupported',
      label: 'Unsupported',
      value: counts.unsupported,
      color: 'bg-unsupported',
    },
  ]

  return (
    <section className="animate-rise rounded-2xl border border-border bg-card/60 p-6 backdrop-blur sm:p-8">
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
            Verification summary
          </p>
          <div className="mt-3 flex items-baseline gap-3">
            <span className="font-mono text-5xl font-medium tabular-nums tracking-tight">
              {Math.round(trust * 100)}%
            </span>
            <span className="text-sm text-muted-foreground">of claims supported</span>
          </div>
        </div>
        <dl className="flex gap-8">
          <div>
            <dt className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
              Claims
            </dt>
            <dd className="mt-1 font-mono text-2xl tabular-nums">{total}</dd>
          </div>
          <div>
            <dt className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
              Avg. confidence
            </dt>
            <dd className="mt-1 font-mono text-2xl tabular-nums">
              {Math.round(avgConfidence * 100)}%
            </dd>
          </div>
        </dl>
      </div>

      {total > 0 && (
        <div className="mt-7 flex h-2 w-full gap-0.5 overflow-hidden rounded-full bg-muted">
          {segments
            .filter((segment) => segment.value > 0)
            .map((segment) => (
              <div
                key={segment.key}
                className={segment.color}
                style={{ width: `${(segment.value / total) * 100}%` }}
                aria-hidden="true"
              />
            ))}
        </div>
      )}

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        {segments.map((segment) => (
          <div
            key={segment.key}
            className="flex items-center justify-between rounded-xl border border-border bg-background/40 px-4 py-3"
          >
            <span className="flex items-center gap-2.5 text-sm text-muted-foreground">
              <span className={`size-2 rounded-full ${segment.color}`} aria-hidden="true" />
              {segment.label}
            </span>
            <span className="font-mono text-lg tabular-nums">{segment.value}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

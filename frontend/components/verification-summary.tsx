'use client'

import { AlertTriangle, CheckCircle2, CircleHelp, FileCheck2 } from 'lucide-react'
import { summarize, type OverallResult, type VerificationResult } from '@/lib/api'

export function VerificationSummary({ results, overall }: { results: VerificationResult[]; overall?: OverallResult | null }) {
  const { counts, total, avgConfidence } = summarize(results)
  const cards = [
    { label: 'Total claims', value: total, icon: FileCheck2, tone: 'text-primary bg-primary/10' },
    { label: 'Supported', value: counts.supported, icon: CheckCircle2, tone: 'text-supported bg-supported/10' },
    { label: 'Contradicted', value: counts.contradicted, icon: AlertTriangle, tone: 'text-contradicted bg-contradicted/10' },
    { label: 'Insufficient evidence', value: counts.insufficient, icon: CircleHelp, tone: 'text-unsupported bg-unsupported/10' },
  ]

  return (
    <section className="rounded-2xl border border-border bg-card p-5 shadow-sm sm:p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Verification overview</p>
          <h2 className="mt-2 text-xl font-semibold tracking-tight">Evidence-backed claim results</h2>
        </div>
        <div className="text-right">
          <p className="text-xs text-muted-foreground">Overall result</p>
          <p className="mt-1 text-2xl font-semibold tabular-nums">{overall?.status || 'INCONCLUSIVE'}</p>
          <p className="text-xs text-muted-foreground">{Math.round((overall?.confidence ?? avgConfidence) * 100)}% confidence</p>
        </div>
      </div>
      {overall?.explanation && <p className="mt-4 text-sm text-muted-foreground">{overall.explanation}</p>}
      <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {cards.map(({ label, value, icon: Icon, tone }) => (
          <div key={label} className="rounded-xl border border-border bg-background/60 p-4">
            <div className={`flex size-9 items-center justify-center rounded-lg ${tone}`}>
              <Icon className="size-4" aria-hidden="true" />
            </div>
            <p className="mt-4 text-2xl font-semibold tabular-nums">{value}</p>
            <p className="mt-1 text-xs font-medium text-muted-foreground">{label}</p>
          </div>
        ))}
      </div>
      {total > 0 && (
        <div className="mt-6">
          <div className="flex h-2 overflow-hidden rounded-full bg-muted">
            {[
              { key: 'supported', value: counts.supported, color: 'bg-supported' },
              { key: 'contradicted', value: counts.contradicted, color: 'bg-contradicted' },
              { key: 'insufficient', value: counts.insufficient, color: 'bg-unsupported' },
            ].map(({ key, value, color }) => (
              <div key={key} className={color} style={{ width: `${(Number(value) / total) * 100}%` }} />
            ))}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Reliability is shown as claim statistics because the API does not return an overall reliability score.
          </p>
        </div>
      )}
    </section>
  )
}

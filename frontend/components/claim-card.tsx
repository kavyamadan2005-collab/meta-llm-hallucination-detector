'use client'

import { useState } from 'react'
import { CheckCircle2, ChevronDown, CircleHelp, ExternalLink, TriangleAlert } from 'lucide-react'
import { formatEvidenceText, formatEvidenceTitle, safeHost, verdictKey, type Evidence, type VerificationResult } from '@/lib/api'
import { cn } from '@/lib/utils'

const styles = {
  supported: { label: 'SUPPORTED', icon: CheckCircle2, text: 'text-supported', chip: 'border-supported/25 bg-supported/10 text-supported', rail: 'bg-supported' },
  contradicted: { label: 'CONTRADICTED', icon: TriangleAlert, text: 'text-contradicted', chip: 'border-contradicted/25 bg-contradicted/10 text-contradicted', rail: 'bg-contradicted' },
  unsupported: { label: 'UNSUPPORTED', icon: CircleHelp, text: 'text-unsupported', chip: 'border-unsupported/25 bg-unsupported/10 text-unsupported', rail: 'bg-unsupported' },
} as const

function EvidenceList({ items, label }: { items: Evidence[]; label: string }) {
  if (!items.length) return null
  return (
    <div className="mt-4 border-t border-border pt-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <div className="mt-3 grid gap-3">
        {items.map((item, index) => {
          const url = item.source_url || item.url
          const score = item.relevance_score ?? item.score
          return (
            <div key={`${label}-${index}`} className="rounded-xl border border-border bg-background/70 p-4">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <p className="text-sm font-semibold">{formatEvidenceTitle(item, index)}</p>
                {typeof score === 'number' && <span className="text-xs tabular-nums text-muted-foreground">Relevance {score.toFixed(2)}</span>}
              </div>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{formatEvidenceText(item) || 'No evidence text returned.'}</p>
              {url && (
                <a href={url} target="_blank" rel="noopener noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-primary hover:underline">
                  {safeHost(url)} <ExternalLink className="size-3" aria-hidden="true" />
                </a>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function ClaimCard({ result, index }: { result: VerificationResult; index: number }) {
  const [open, setOpen] = useState(false)
  const kind = verdictKey(result.verdict)
  const style = styles[kind]
  const Icon = style.icon
  const evidence = Array.isArray(result.evidence) ? result.evidence : []
  const supporting = Array.isArray(result.supporting_evidence) ? result.supporting_evidence : []
  const contradicting = Array.isArray(result.contradicting_evidence) ? result.contradicting_evidence : []
  const confidence = Math.round((Number(result.confidence) || 0) * 100)

  return (
    <article className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm" style={{ animationDelay: `${Math.min(index, 8) * 50}ms` }}>
      <div className="flex">
        <div className={cn('w-1 shrink-0', style.rail)} />
        <div className="min-w-0 flex-1 p-5 sm:p-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-md bg-muted px-2 py-1 font-mono text-xs font-semibold text-muted-foreground">{result.claim_id}</span>
            <span className={cn('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-bold tracking-wide', style.chip)}>
              <Icon className="size-3.5" aria-hidden="true" /> {style.label}
            </span>
            <span className="ml-auto text-sm font-semibold tabular-nums text-muted-foreground">{confidence}% confidence</span>
          </div>
          <p className="mt-4 text-base font-medium leading-relaxed">{result.claim_text}</p>
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-muted">
            <div className={cn('h-full rounded-full', style.rail)} style={{ width: `${confidence}%` }} />
          </div>
          {result.explanation && <p className="mt-4 border-l-2 border-primary/30 pl-4 text-sm leading-relaxed text-muted-foreground">{result.explanation}</p>}
          <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open} className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-primary hover:text-primary/80">
            <ChevronDown className={cn('size-4 transition-transform', open && 'rotate-180')} />
            {evidence.length ? `${evidence.length} evidence ${evidence.length === 1 ? 'source' : 'sources'}` : 'No evidence returned'}
          </button>
          {open && (
            <div>
              {supporting.length || contradicting.length ? (
                <>
                  <EvidenceList items={supporting} label="Supporting evidence" />
                  <EvidenceList items={contradicting} label="Contradicting evidence" />
                </>
              ) : (
                <EvidenceList items={evidence} label="Retrieved evidence" />
              )}
              {!evidence.length && <p className="mt-4 rounded-lg bg-muted p-3 text-sm text-muted-foreground">No evidence was found for this claim.</p>}
            </div>
          )}
        </div>
      </div>
    </article>
  )
}

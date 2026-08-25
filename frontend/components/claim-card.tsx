'use client'

import { useState } from 'react'
import { CheckCircle2, ChevronDown, CircleAlert, ExternalLink, XCircle } from 'lucide-react'
import {
  formatEvidenceText,
  formatEvidenceTitle,
  safeHost,
  verdictKey,
  type VerificationResult,
} from '@/lib/api'
import { cn } from '@/lib/utils'

const VERDICT_STYLES = {
  supported: {
    label: 'Supported',
    icon: CheckCircle2,
    text: 'text-supported',
    chip: 'border-supported/30 bg-supported/10 text-supported',
    bar: 'bg-supported',
    rail: 'bg-supported',
  },
  contradicted: {
    label: 'Contradicted',
    icon: XCircle,
    text: 'text-contradicted',
    chip: 'border-contradicted/30 bg-contradicted/10 text-contradicted',
    bar: 'bg-contradicted',
    rail: 'bg-contradicted',
  },
  unsupported: {
    label: 'Unsupported',
    icon: CircleAlert,
    text: 'text-unsupported',
    chip: 'border-unsupported/30 bg-unsupported/10 text-unsupported',
    bar: 'bg-unsupported',
    rail: 'bg-unsupported',
  },
} as const

export function ClaimCard({ result, index }: { result: VerificationResult; index: number }) {
  const evidence = Array.isArray(result.evidence) ? result.evidence : []
  const [open, setOpen] = useState(false)
  const style = VERDICT_STYLES[verdictKey(result.verdict)]
  const Icon = style.icon
  const confidence = Math.round((Number(result.confidence) || 0) * 100)

  return (
    <article
      className="animate-rise overflow-hidden rounded-2xl border border-border bg-card/60 backdrop-blur"
      style={{ animationDelay: `${Math.min(index, 8) * 60}ms` }}
    >
      <div className="flex">
        <div className={cn('w-0.5 shrink-0', style.rail)} aria-hidden="true" />
        <div className="min-w-0 flex-1 p-5 sm:p-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
              {result.claim_id}
            </span>
            <span
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] uppercase tracking-wide',
                style.chip,
              )}
            >
              <Icon className="size-3.5" aria-hidden="true" />
              {style.label}
            </span>
            <span className="ml-auto font-mono text-sm tabular-nums text-muted-foreground">
              {confidence}% confidence
            </span>
          </div>

          <p className="mt-4 text-pretty text-[15px] leading-relaxed">{result.claim_text}</p>

          <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn('h-1 rounded-full transition-[width] duration-700', style.bar)}
              style={{ width: `${confidence}%` }}
              role="progressbar"
              aria-valuenow={confidence}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`Confidence for claim ${result.claim_id}`}
            />
          </div>

          {result.explanation && (
            <p className="mt-5 border-l border-border pl-4 text-sm leading-relaxed text-muted-foreground">
              {result.explanation}
            </p>
          )}

          {evidence.length > 0 && (
            <div className="mt-5">
              <button
                type="button"
                onClick={() => setOpen((value) => !value)}
                aria-expanded={open}
                className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wide text-muted-foreground transition-colors hover:text-foreground"
              >
                <ChevronDown
                  className={cn('size-3.5 transition-transform', open && 'rotate-180')}
                  aria-hidden="true"
                />
                {evidence.length} evidence {evidence.length === 1 ? 'source' : 'sources'}
              </button>

              {open && (
                <ul className="mt-4 grid gap-3">
                  {evidence.map((item, evidenceIndex) => {
                    const url =
                      (typeof item.source_url === 'string' && item.source_url) ||
                      (typeof item.url === 'string' && item.url) ||
                      null
                    const text = formatEvidenceText(item)
                    const score =
                      typeof item.relevance_score === 'number'
                        ? item.relevance_score
                        : typeof item.score === 'number'
                          ? item.score
                          : null
                    return (
                      <li
                        key={`${result.claim_id}-evidence-${evidenceIndex}`}
                        className="rounded-xl border border-border bg-background/40 p-4"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <p className="text-sm font-medium leading-snug">
                            {formatEvidenceTitle(item, evidenceIndex)}
                          </p>
                          {typeof score === 'number' && (
                            <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                              score {score.toFixed(2)}
                            </span>
                          )}
                        </div>
                        {text && (
                          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                            {text}
                          </p>
                        )}
                        {url && (
                          <a
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 font-mono text-[11px] text-primary transition-colors hover:border-primary/50 hover:bg-primary/10"
                          >
                            {safeHost(url)}
                            <ExternalLink className="size-3" aria-hidden="true" />
                          </a>
                        )}
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </article>
  )
}

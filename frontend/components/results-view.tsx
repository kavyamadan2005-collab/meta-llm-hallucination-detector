'use client'

import { Cpu, MessageSquareQuote, RotateCcw } from 'lucide-react'
import type { AnalyzeResponse } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { ClaimCard } from '@/components/claim-card'
import { VerificationSummary } from '@/components/verification-summary'

export function ResultsView({ data, onReset }: { data: AnalyzeResponse; onReset: () => void }) {
  const results = Array.isArray(data.verification_results) ? data.verification_results : []
  const claimCount = Array.isArray(data.claims) ? data.claims.length : results.length
  const model = data.llm_response?.model

  return (
    <div className="space-y-6">
      <section className="animate-rise rounded-2xl border border-border bg-card/60 p-6 backdrop-blur sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <p className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
              <MessageSquareQuote className="size-3.5" aria-hidden="true" />
              Question
            </p>
            <h2 className="mt-3 text-pretty text-2xl font-medium leading-snug sm:text-3xl">
              {data.question}
            </h2>
          </div>
          <Button variant="ghost" onClick={onReset} className="gap-2 text-muted-foreground">
            <RotateCcw className="size-4" aria-hidden="true" />
            New question
          </Button>
        </div>

        <div className="mt-8 rounded-xl border border-border bg-background/40 p-5 sm:p-6">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-border pb-4">
            <p className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wide text-primary">
              <Cpu className="size-3.5" aria-hidden="true" />
              Generated answer
            </p>
            {model && (
              <p className="font-mono text-[11px] text-muted-foreground">
                model <span className="text-foreground">{model}</span>
              </p>
            )}
            <p className="font-mono text-[11px] text-muted-foreground">
              claims extracted <span className="text-foreground">{claimCount}</span>
            </p>
          </div>
          <p className="mt-4 whitespace-pre-wrap text-pretty text-[15px] leading-relaxed text-foreground/90">
            {data.llm_response?.response_text}
          </p>
        </div>
      </section>

      {results.length > 0 ? (
        <>
          <VerificationSummary results={results} />

          <section>
            <div className="flex items-baseline justify-between px-1">
              <h3 className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                Claim-by-claim verification
              </h3>
              <span className="font-mono text-[11px] tabular-nums text-muted-foreground/70">
                {results.length} {results.length === 1 ? 'claim' : 'claims'}
              </span>
            </div>
            <div className="mt-4 grid gap-4">
              {results.map((result, index) => (
                <ClaimCard
                  key={`${result.claim_id}-${index}`}
                  result={result}
                  index={index}
                />
              ))}
            </div>
          </section>
        </>
      ) : (
        <p className="rounded-2xl border border-border bg-card/60 p-6 text-sm text-muted-foreground">
          The backend returned no verification results for this answer.
        </p>
      )}
    </div>
  )
}

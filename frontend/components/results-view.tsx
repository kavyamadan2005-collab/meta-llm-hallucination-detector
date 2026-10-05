'use client'

import { useMemo, useState } from 'react'
import { ChevronDown, Clock3, Cpu, Network, RotateCcw, Search, ShieldCheck } from 'lucide-react'
import type { AnalyzeResponse, VerificationResult } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { ClaimCard } from '@/components/claim-card'
import { VerificationSummary } from '@/components/verification-summary'

function GraphPanel({ graph }: { graph: Record<string, unknown> | null | undefined }) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes as Record<string, unknown>[] : []
  const edges = Array.isArray(graph?.edges) ? graph.edges as Record<string, unknown>[] : []
  const claims = Array.isArray(graph?.claims) ? graph.claims as Record<string, unknown>[] : []
  if (!nodes.length && !edges.length) {
    return <div className="rounded-xl border border-dashed border-border p-5 text-sm text-muted-foreground">No claim graph data was returned for this analysis.</div>
  }
  return (
    <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
      <div className="grid gap-3 sm:grid-cols-2">
        {nodes.map((node, index) => (
          <div key={String(node.id ?? index)} className="rounded-xl border border-border bg-background/70 p-4">
            <p className="text-sm font-semibold">{String(node.text ?? node.id ?? 'Entity')}</p>
            <p className="mt-1 text-xs uppercase tracking-wide text-primary">{String(node.entity_type ?? 'entity')}</p>
            {Array.isArray(node.claim_ids) && <p className="mt-3 text-xs text-muted-foreground">Claims: {node.claim_ids.join(', ')}</p>}
          </div>
        ))}
      </div>
      <div className="rounded-xl border border-border bg-background/70 p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Relationships</p>
        <div className="mt-3 space-y-3">
          {edges.length ? edges.map((edge, index) => (
            <div key={index} className="rounded-lg bg-muted/70 p-3 text-sm">
              <span className="font-medium">{String(edge.source)}</span>
              <span className="mx-2 text-primary">→</span>
              <span className="font-medium">{String(edge.target)}</span>
              <p className="mt-1 text-xs text-muted-foreground">{String(edge.relationship_type ?? 'related')}</p>
            </div>
          )) : <p className="text-sm text-muted-foreground">No relationships returned.</p>}
        </div>
        {claims.length > 0 && <p className="mt-4 text-xs text-muted-foreground">{claims.length} claim associations preserved in the graph.</p>}
      </div>
    </div>
  )
}

function Pipeline() {
  const steps = [
    ['01', 'Question', 'Input'],
    ['02', 'LLM response', 'Generated'],
    ['03', 'Claims', 'Extracted'],
    ['04', 'Evidence', 'Retrieved'],
    ['05', 'Verification', 'Complete'],
  ]
  return (
    <div className="grid gap-2 sm:grid-cols-5">
      {steps.map(([number, label, state], index) => (
        <div key={label} className="relative rounded-xl border border-border bg-card p-3 shadow-sm">
          <p className="text-[10px] font-bold tracking-[0.16em] text-primary">{number}</p>
          <p className="mt-3 text-sm font-semibold">{label}</p>
          <p className="mt-1 text-xs text-muted-foreground">{state}</p>
          {index < steps.length - 1 && <span className="absolute -right-2 top-1/2 z-10 hidden text-primary sm:block">→</span>}
        </div>
      ))}
    </div>
  )
}

export function ResultsView({ data, onReset }: { data: AnalyzeResponse; onReset: () => void }) {
  const results = Array.isArray(data.verification_results) ? data.verification_results : []
  const claimCount = Array.isArray(data.claims) ? data.claims.length : results.length
  const [graphOpen, setGraphOpen] = useState(false)
  const timestamp = data.llm_response?.timestamp
  const formattedTimestamp = useMemo(() => {
    if (!timestamp) return null
    const date = new Date(timestamp)
    return Number.isNaN(date.getTime()) ? timestamp : date.toLocaleString()
  }, [timestamp])

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-border bg-card p-6 shadow-sm sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Analysis result</p>
            <h1 className="mt-3 text-2xl font-semibold tracking-tight sm:text-3xl">{data.question}</h1>
          </div>
          <Button variant="outline" onClick={onReset} className="gap-2">
            <RotateCcw className="size-4" /> New analysis
          </Button>
        </div>
        <div className="mt-7 grid gap-4 lg:grid-cols-[1.5fr_0.5fr]">
          <div className="rounded-xl border border-primary/15 bg-primary/[0.04] p-5">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-primary"><Cpu className="size-4" /> Generated answer</div>
            <p className="mt-4 whitespace-pre-wrap text-[15px] leading-7">{data.llm_response.response_text}</p>
          </div>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
            <div className="rounded-xl border border-border bg-background/60 p-4"><p className="text-xs text-muted-foreground">Model</p><p className="mt-2 break-words text-sm font-semibold">{data.llm_response.model || 'Not provided'}</p></div>
            <div className="rounded-xl border border-border bg-background/60 p-4"><p className="text-xs text-muted-foreground">Claims extracted</p><p className="mt-2 text-2xl font-semibold">{claimCount}</p></div>
            {formattedTimestamp && <div className="col-span-2 flex items-center gap-2 text-xs text-muted-foreground lg:col-span-1"><Clock3 className="size-3.5" /> {formattedTimestamp}</div>}
          </div>
        </div>
      </section>

      <Pipeline />
      <VerificationSummary results={results} />

      <section>
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div><p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Claim verification</p><h2 className="mt-2 text-xl font-semibold">Review each extracted claim</h2></div>
          <span className="text-sm text-muted-foreground">{results.length} result{results.length === 1 ? '' : 's'}</span>
        </div>
        {results.length ? (
          <div className="mt-4 grid gap-4">{results.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} />)}</div>
        ) : (
          <div className="mt-4 rounded-2xl border border-dashed border-border bg-card p-6 text-sm text-muted-foreground">No claims were extracted or no verification results were returned.</div>
        )}
      </section>

      <section className="rounded-2xl border border-border bg-card shadow-sm">
        <button type="button" onClick={() => setGraphOpen((value) => !value)} aria-expanded={graphOpen} className="flex w-full items-center justify-between p-5 text-left sm:p-6">
          <span className="flex items-center gap-3"><span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary"><Network className="size-4" /></span><span><span className="block text-xs font-semibold uppercase tracking-[0.16em] text-primary">Claim graph</span><span className="mt-1 block text-sm text-muted-foreground">Entities and relationships returned by the backend</span></span></span>
          <ChevronDown className={`size-5 text-muted-foreground transition-transform ${graphOpen ? 'rotate-180' : ''}`} />
        </button>
        {graphOpen && <div className="border-t border-border p-5 sm:p-6"><GraphPanel graph={data.graph} /></div>}
      </section>

      <section className="rounded-2xl border border-primary/15 bg-primary/[0.04] p-5 sm:p-6">
        <div className="flex items-start gap-3"><ShieldCheck className="mt-0.5 size-5 shrink-0 text-primary" /><div><p className="text-sm font-semibold">Analysis summary</p><p className="mt-2 text-sm leading-relaxed text-muted-foreground">The response above was decomposed into {claimCount} claim{claimCount === 1 ? '' : 's'} and evaluated using the verdicts and evidence returned by the analysis pipeline. Expand a claim to inspect its sources.</p></div></div>
      </section>
    </div>
  )
}

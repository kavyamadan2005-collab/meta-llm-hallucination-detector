'use client'

import { useMemo, useRef, useState } from 'react'
import { ArrowLeft, CheckCircle2, CircleHelp, ExternalLink, Maximize2, Minus, Plus, SearchCheck, TriangleAlert } from 'lucide-react'
import { formatEvidenceText, formatEvidenceTitle, safeHost, verdictKey, type Evidence, type VerificationResult } from '@/lib/api'
import { cn } from '@/lib/utils'

const verdictStyles = {
  supported: {
    label: 'SUPPORTED',
    description: 'Supported',
    icon: CheckCircle2,
    text: 'text-supported',
    border: 'border-supported/35',
    background: 'bg-supported/10',
    dot: 'bg-supported',
  },
  contradicted: {
    label: 'CONTRADICTED',
    description: 'Contradicted',
    icon: TriangleAlert,
    text: 'text-contradicted',
    border: 'border-contradicted/35',
    background: 'bg-contradicted/10',
    dot: 'bg-contradicted',
  },
  insufficient: {
    label: 'INSUFFICIENT EVIDENCE',
    description: 'Insufficient',
    icon: CircleHelp,
    text: 'text-unsupported',
    border: 'border-unsupported/35',
    background: 'bg-unsupported/10',
    dot: 'bg-unsupported',
  },
} as const

function EvidenceDetails({ result }: { result: VerificationResult }) {
  const evidence = [
    ...(result.contradicting_evidence || []),
    ...(result.supporting_evidence || []),
    ...(result.evidence || []),
  ].filter((item, index, items) => items.findIndex((candidate) => (
    formatEvidenceText(candidate) === formatEvidenceText(item) && (candidate.source_url || candidate.url) === (item.source_url || item.url)
  )) === index)

  if (!evidence.length) {
    return <p className="mt-2 text-sm text-muted-foreground">No evidence was found for this claim.</p>
  }

  return (
    <div className="mt-3 space-y-3">
      {evidence.slice(0, 3).map((item, index) => {
        const url = item.source_url || item.url
        return (
          <div key={`${formatEvidenceTitle(item, index)}-${index}`} className="rounded-xl border border-border bg-muted/40 p-3">
            <p className="text-sm font-semibold">{formatEvidenceTitle(item, index)}</p>
            <p className="mt-1 text-sm leading-5 text-muted-foreground">{formatEvidenceText(item) || 'No evidence text returned.'}</p>
            <p className="mt-2 text-xs text-muted-foreground"><span className="font-semibold text-foreground">Source:</span> {item.source_title || item.title || item.source || (url ? safeHost(url) : 'Source not provided')}</p>
            {url && <a href={url} target="_blank" rel="noopener noreferrer" className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline">{safeHost(url)} <ExternalLink className="size-3" /></a>}
          </div>
        )
      })}
    </div>
  )
}

export function ClaimGraph({ answer, results, standalone = false, onBack }: { answer: string; results: VerificationResult[]; standalone?: boolean; onBack?: () => void }) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const dragStart = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null)

  const counts = useMemo(() => results.reduce((value, result) => {
    const key = verdictKey(result.verdict)
    value[key] += 1
    return value
  }, { supported: 0, contradicted: 0, insufficient: 0 }), [results])

  const visibleResults = results
  const selected = visibleResults.find((result) => result.claim_id === selectedId) || results.find((result) => result.claim_id === selectedId)

  if (!results.length) {
    return <div className="rounded-2xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground">No verifiable claims were extracted from this answer.</div>
  }

  return (
    <div className={cn('flex min-h-0 flex-col', standalone ? 'h-full overflow-hidden' : 'space-y-4')}>
      {!standalone && <div className="grid gap-3 rounded-2xl border border-border bg-card p-4 sm:grid-cols-[auto_1fr] sm:items-center">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Verification summary</p>
          <p className="mt-1 text-lg font-semibold">Claims: {results.length}</p>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm font-semibold">
          <span className="text-supported">● Supported: {counts.supported}</span>
          <span className="text-contradicted">● Contradicted: {counts.contradicted}</span>
          <span className="text-unsupported">● Insufficient: {counts.insufficient}</span>
        </div>
      </div>}

      <div className={cn('grid min-h-0 gap-5 lg:grid-cols-[minmax(0,1fr)_320px]', standalone ? 'h-full flex-1' : 'lg:h-[calc(100dvh-13rem)]')}>
        <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-slate-700 bg-slate-950 shadow-xl shadow-slate-950/10">
          <div className="flex items-center border-b border-slate-800 px-5 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-white"><SearchCheck className="size-4 text-cyan-300" /> Answer → claims → verdicts</div>
          </div>

          <div
            className="relative min-h-0 flex-1 overflow-auto"
            onWheel={(event) => {
              if (!event.ctrlKey) return
              event.preventDefault()
              setZoom((value) => Math.min(1.5, Math.max(0.7, value - event.deltaY * 0.001)))
            }}
            onPointerMove={(event) => {
              if (!dragStart.current) return
              setPan({
                x: dragStart.current.panX + event.clientX - dragStart.current.x,
                y: dragStart.current.panY + event.clientY - dragStart.current.y,
              })
            }}
            onPointerUp={() => { dragStart.current = null }}
            onPointerCancel={() => { dragStart.current = null }}
          >
            {standalone && onBack && <button type="button" onClick={onBack} aria-label="Back to analysis" title="Back to analysis" className="group absolute left-4 top-3 z-10 flex size-8 items-center justify-center rounded-full border border-transparent bg-transparent text-slate-400/35 transition-all hover:border-slate-600 hover:bg-slate-900 hover:text-slate-200 hover:shadow-sm focus-visible:border-slate-500 focus-visible:bg-slate-900 focus-visible:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/40">
              <ArrowLeft className="size-4 transition-transform group-hover:-translate-x-0.5" />
            </button>}
            <div className="absolute right-4 top-3 z-10 flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-950/90 p-1 shadow-lg">
              <button type="button" onClick={() => setZoom((value) => Math.max(0.7, value - 0.1))} aria-label="Zoom out" className="rounded p-1.5 text-slate-300 hover:bg-slate-800 hover:text-white"><Minus className="size-3.5" /></button>
              <span className="min-w-10 text-center text-[10px] font-semibold text-slate-400">{Math.round(zoom * 100)}%</span>
              <button type="button" onClick={() => setZoom((value) => Math.min(1.5, value + 0.1))} aria-label="Zoom in" className="rounded p-1.5 text-slate-300 hover:bg-slate-800 hover:text-white"><Plus className="size-3.5" /></button>
              <button type="button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }} aria-label="Fit graph to view" title="Fit graph to view" className="rounded p-1.5 text-slate-300 hover:bg-slate-800 hover:text-white"><Maximize2 className="size-3.5" /></button>
            </div>
            <div
              className="grid-backdrop min-h-full w-full cursor-grab px-4 pb-8 pt-6 sm:px-6 active:cursor-grabbing"
              onPointerDown={(event) => {
                if (event.button !== 0 || event.target !== event.currentTarget) return
                dragStart.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y }
              }}
            >
              <div className="mx-auto flex w-full max-w-[1400px] flex-col items-center transition-transform duration-150" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: 'top center' }}>
                <div className="mx-auto max-w-[430px] rounded-2xl border border-cyan-300/40 bg-cyan-950/70 p-5 text-center text-white shadow-lg shadow-cyan-950/30">
                  <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-200">Original answer</p>
                  <p className="mt-2 text-lg font-bold">LLM Answer</p>
                  <p className="mt-2 line-clamp-3 text-sm leading-5 text-slate-300">{answer}</p>
                </div>
                <div className="flex h-8 w-0 flex-col items-center border-l border-dashed border-cyan-400/70"><span className="mt-auto text-cyan-300">↓</span></div>
                <div className="rounded-xl border border-cyan-300/25 bg-slate-900/80 px-6 py-2.5 text-center text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-100 shadow-sm">
                  Claim decomposition
                </div>
                <div className="flex h-8 w-0 flex-col items-center border-l border-dashed border-cyan-400/70"><span className="mt-auto text-cyan-300">↓</span></div>
                {visibleResults.length ? (
                  <div
                    className={cn(
                      'mx-auto grid w-full items-start justify-items-center gap-x-5 gap-y-6',
                      visibleResults.length === 1 && 'grid-cols-1',
                      visibleResults.length === 2 && 'grid-cols-1 sm:grid-cols-2',
                      visibleResults.length === 3 && 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
                      visibleResults.length >= 4 && 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
                    )}
                    style={{
                      maxWidth: `${Math.min(visibleResults.length, 4) * 280}px`,
                    }}
                  >
                    {visibleResults.map((result, index) => {
                      const style = verdictStyles[verdictKey(result.verdict)]
                      const Icon = style.icon
                      const isSelected = result.claim_id === selectedId
                      return (
                        <div key={`${result.claim_id}-${index}`} className="flex w-full max-w-[260px] min-w-0 flex-col items-center">
                          <button type="button" onClick={() => setSelectedId(result.claim_id)} className={cn('w-full rounded-xl border bg-slate-900 p-3 text-left transition-all hover:-translate-y-0.5 hover:border-cyan-300/70', isSelected ? 'border-cyan-200 ring-2 ring-cyan-300/30' : 'border-slate-700')}>
                            <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-200">Claim {index + 1}</p>
                            <p className="mt-2 min-h-[44px] line-clamp-3 text-xs font-semibold leading-4 text-white">{result.claim_text}</p>
                            <p className={cn('mt-2 text-[10px] font-bold uppercase tracking-wide', style.text)}>{style.description}</p>
                          </button>
                          <div className="flex h-7 w-0 flex-col items-center border-l border-dashed border-slate-500"><span className={cn('mt-auto text-xs', style.text)}>↓</span></div>
                          <button type="button" onClick={() => setSelectedId(result.claim_id)} className={cn('flex w-full items-center justify-center gap-1.5 rounded-xl border p-2.5 text-center transition-colors hover:brightness-110', style.border, style.background, style.text)}>
                            <Icon className="size-3.5" /><span className="text-[10px] font-bold tracking-wide">{style.label}</span>
                          </button>
                        </div>
                      )
                    })}
                  </div>
                ) : <p className="rounded-xl border border-dashed border-slate-700 p-8 text-center text-sm text-slate-400">No claims match this filter.</p>}
              </div>
            </div>
          </div>
        </section>

        <aside className="min-h-0 overflow-y-auto overscroll-contain rounded-2xl border border-border bg-card p-5 lg:h-full">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Graph inspector</p>
          {selected ? (() => {
            const style = verdictStyles[verdictKey(selected.verdict)]
            return <div className="mt-4">
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-muted-foreground">{selected.claim_id}</p>
              <h3 className="mt-2 text-lg font-semibold">Claim details</h3>
              <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Full claim</p>
              <p className="mt-1 text-sm leading-6">{selected.claim_text}</p>
              <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Status</p>
              <p className={cn('mt-1 flex items-center gap-2 text-sm font-bold', style.text)}><span className={cn('size-2 rounded-full', style.dot)} />{style.label}</p>
              <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Evidence</p>
              <EvidenceDetails result={selected} />
              <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Explanation</p>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">{selected.explanation || 'No explanation was returned for this claim.'}</p>
              {typeof selected.confidence === 'number' && <p className="mt-4 text-xs text-muted-foreground">Confidence: {Math.round(selected.confidence * 100)}%</p>}
            </div>
          })() : <div className="mt-5 text-sm leading-6 text-muted-foreground"><p className="font-semibold text-foreground">Select a claim or verdict</p><p className="mt-2">Evidence, source details, explanation, and confidence will appear here without crowding the graph.</p></div>}
        </aside>
      </div>
    </div>
  )
}

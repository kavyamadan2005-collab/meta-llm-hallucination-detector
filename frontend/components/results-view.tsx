'use client'

import { useMemo, useState } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Clock3,
  Cpu,
  Focus,
  Network,
  RotateCcw,
  ShieldCheck,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import type { AnalyzeResponse, VerificationResult } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { ClaimCard } from '@/components/claim-card'
import { VerificationSummary } from '@/components/verification-summary'
import { cn } from '@/lib/utils'

export type ResearchStage = 0 | 1 | 2 | 3 | 4 | 5

const STAGES = ['Answer', 'Claims', 'Evidence', 'Verification', 'Knowledge graph', 'Summary']

function StageNavigation({
  stage,
  onStageChange,
}: {
  stage: ResearchStage
  onStageChange: (stage: ResearchStage) => void
}) {
  return (
    <nav aria-label="Analysis stages" className="border-b border-border pb-5">
      <div className="flex items-center gap-1 overflow-x-auto pb-1">
        {STAGES.map((label, index) => (
          <button
            key={label}
            type="button"
            onClick={() => onStageChange(index as ResearchStage)}
            className={cn(
              'shrink-0 rounded-full px-3 py-1.5 text-xs font-semibold transition-colors',
              index === stage
                ? 'bg-primary text-primary-foreground'
                : index < stage
                  ? 'bg-primary/10 text-primary hover:bg-primary/15'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
            )}
          >
            {index < stage && <Check className="mr-1 inline size-3" aria-hidden="true" />}
            {label}
          </button>
        ))}
      </div>
    </nav>
  )
}

function StepButtons({
  stage,
  onStageChange,
}: {
  stage: ResearchStage
  onStageChange: (stage: ResearchStage) => void
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-border pt-5">
      {stage > 0 ? (
        <Button variant="outline" onClick={() => onStageChange((stage - 1) as ResearchStage)} className="gap-2">
          <ArrowLeft className="size-4" /> Back
        </Button>
      ) : <span />}
      {stage < 5 && (
        <Button onClick={() => onStageChange((stage + 1) as ResearchStage)} className="gap-2">
          Next <ArrowRight className="size-4" />
        </Button>
      )}
    </div>
  )
}

type GraphItem = { id: string; label: string; kind: 'entity' | 'claim'; x: number; y: number; detail?: string; status?: string }
type GraphLink = { source: string; target: string; label?: string }

function GraphPanel({
  graph,
  results,
  claims,
}: {
  graph: Record<string, unknown> | null | undefined
  results: VerificationResult[]
  claims: AnalyzeResponse['claims']
}) {
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [nodeOffsets, setNodeOffsets] = useState<Record<string, { x: number; y: number }>>({})
  const [selected, setSelected] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes as Record<string, unknown>[] : []
  const rawClaims = Array.isArray(graph?.claims) ? graph.claims as Record<string, unknown>[] : []
  const claimById = new Map(results.map((result) => [result.claim_id, result]))
  const claimTextById = new Map((claims || []).map((claim) => [claim.claim_id, claim.claim_text]))
  const items = useMemo<GraphItem[]>(() => {
    const entities: GraphItem[] = nodes.map((node, index) => ({
      id: String(node.id ?? `entity-${index}`),
      label: String(node.text ?? node.id ?? 'Entity'),
      kind: 'entity',
      x: 140 + (index % 3) * 230,
      y: 120 + Math.floor(index / 3) * 150,
      detail: String(node.entity_type ?? 'entity'),
    }))
    const ids = new Set(results.map((result) => result.claim_id))
    rawClaims.forEach((claim) => ids.add(String(claim.claim_id ?? claim.id ?? '')))
    nodes.forEach((node) => {
      if (Array.isArray(node.claim_ids)) node.claim_ids.forEach((id) => ids.add(String(id)))
    })
    const claimItems = [...ids].filter(Boolean).map((id, index) => {
      const result = claimById.get(id)
      return {
        id,
        label: id,
        kind: 'claim' as const,
        x: 120 + (index % 4) * 180,
        y: 430 + Math.floor(index / 4) * 120,
        detail: result?.claim_text || claimTextById.get(id) || 'Claim text unavailable',
        status: result?.verdict,
      }
    })
    return [...entities, ...claimItems]
  }, [claimById, claimTextById, nodes, rawClaims, results])
  const links = useMemo<GraphLink[]>(() => {
    const result: GraphLink[] = []
    const entityIds = new Set(nodes.map((node, index) => String(node.id ?? `entity-${index}`)))
    nodes.forEach((node, index) => {
      const source = String(node.id ?? `entity-${index}`)
      if (Array.isArray(node.claim_ids)) node.claim_ids.forEach((id) => result.push({ source, target: String(id), label: 'claim association' }))
    })
    if (Array.isArray(graph?.edges)) {
      ;(graph.edges as Record<string, unknown>[]).forEach((edge) => {
        const source = String(edge.source ?? '')
        const target = String(edge.target ?? '')
        if (source && target && (entityIds.has(source) || items.some((item) => item.id === source))) {
          result.push({ source, target, label: String(edge.relationship_type ?? 'relationship') })
        }
      })
    }
    return result
  }, [graph, items, nodes])
  const selectedItem = items.find((item) => item.id === selected)
  const connected = new Set(selected ? links.flatMap((link) => link.source === selected ? [link.target] : link.target === selected ? [link.source] : []) : [])
  const hasLinks = links.length > 0
  const positionOf = (item: GraphItem) => {
    const adjustment = nodeOffsets[item.id] || { x: 0, y: 0 }
    return { x: item.x + adjustment.x, y: item.y + adjustment.y }
  }

  if (!items.length) {
    return <div className="rounded-xl border border-dashed border-border p-6 text-sm text-muted-foreground">No claim graph data was returned for this analysis.</div>
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
      <div
        className="relative min-h-[520px] overflow-hidden rounded-2xl border border-slate-700 bg-slate-950"
        onPointerMove={(event) => {
          if (draggedNode) {
            setNodeOffsets((value) => ({
              ...value,
              [draggedNode]: {
                x: (value[draggedNode]?.x || 0) + event.movementX / zoom,
                y: (value[draggedNode]?.y || 0) + event.movementY / zoom,
              },
            }))
          } else if (dragging) {
            setOffset((value) => ({ x: value.x + event.movementX, y: value.y + event.movementY }))
          }
        }}
        onPointerUp={() => { setDragging(false); setDraggedNode(null) }}
        onPointerLeave={() => { setDragging(false); setDraggedNode(null) }}
      >
        <div className="absolute left-4 top-4 z-10 flex gap-1 rounded-lg border border-slate-700 bg-slate-900/90 p-1">
          <button type="button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(2, value + 0.15))} className="rounded p-2 text-slate-300 hover:bg-slate-700"><ZoomIn className="size-4" /></button>
          <button type="button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.6, value - 0.15))} className="rounded p-2 text-slate-300 hover:bg-slate-700"><ZoomOut className="size-4" /></button>
          <button type="button" aria-label="Reset graph view" onClick={() => { setZoom(1); setOffset({ x: 0, y: 0 }); setNodeOffsets({}) }} className="rounded p-2 text-slate-300 hover:bg-slate-700"><Focus className="size-4" /></button>
        </div>
        <svg
          viewBox="0 0 860 620"
          className="h-full min-h-[520px] w-full cursor-grab active:cursor-grabbing"
          onPointerDown={() => setDragging(true)}
          role="img"
          aria-label="Interactive claim knowledge graph"
          style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}
        >
          <defs><pattern id="graph-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M 32 0 L 0 0 0 32" fill="none" stroke="#334155" strokeOpacity=".35" /></pattern></defs>
          <rect width="860" height="620" fill="url(#graph-grid)" />
          {hasLinks ? links.map((link, index) => {
            const source = items.find((item) => item.id === link.source)
            const target = items.find((item) => item.id === link.target)
            if (!source || !target) return null
            const sourcePosition = positionOf(source)
            const targetPosition = positionOf(target)
            const quiet = selected && source.id !== selected && target.id !== selected
            return <g key={`${link.source}-${link.target}-${index}`} opacity={quiet ? 0.18 : 0.8}><line x1={sourcePosition.x} y1={sourcePosition.y} x2={targetPosition.x} y2={targetPosition.y} stroke="#64748b" strokeWidth="1.5" /><text x={(sourcePosition.x + targetPosition.x) / 2} y={(sourcePosition.y + targetPosition.y) / 2 - 6} fill="#94a3b8" fontSize="10" textAnchor="middle">{link.label}</text></g>
          }) : <text x="430" y="300" fill="#94a3b8" fontSize="14" textAnchor="middle">No explicit relationships returned</text>}
          {items.map((item) => {
            const active = item.id === selected
            const related = !selected || active || connected.has(item.id)
            const fill = item.kind === 'entity' ? '#0e7490' : item.status ? (item.status.toUpperCase().includes('SUPPORT') ? '#158f78' : item.status.toUpperCase().includes('CONTRADICT') ? '#b94a48' : '#a87822') : '#475569'
            const position = positionOf(item)
            return <g key={item.id} opacity={related ? 1 : 0.25} onPointerDown={(event) => { event.stopPropagation(); setSelected(item.id); setDraggedNode(item.id) }} className="cursor-pointer">
              <circle cx={position.x} cy={position.y} r={active ? 27 : 22} fill={fill} stroke={active ? '#e2e8f0' : '#94a3b8'} strokeWidth={active ? 3 : 1.5} />
              <text x={position.x} y={position.y + 4} fill="white" fontSize="12" fontWeight="600" textAnchor="middle">{item.label.slice(0, 12)}</text>
            </g>
          })}
        </svg>
      </div>
      <aside className="rounded-2xl border border-border bg-card p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Selected item</p>
        {selectedItem ? <><h3 className="mt-3 text-lg font-semibold">{selectedItem.label}</h3><p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">{selectedItem.kind} · {selectedItem.detail}</p>{selectedItem.status && <p className="mt-4 text-sm font-semibold text-primary">{selectedItem.status} · {Math.round((claimById.get(selectedItem.id)?.confidence || 0) * 100)}% confidence</p>}<p className="mt-4 text-sm leading-6 text-muted-foreground">{selectedItem.kind === 'claim' ? selectedItem.detail : 'Click connected nodes to trace this entity through the analysis.'}</p></> : <p className="mt-3 text-sm leading-6 text-muted-foreground">Select an entity or claim node to inspect its connections and details.</p>}
        <div className="mt-6 space-y-2 border-t border-border pt-4 text-xs text-muted-foreground"><p><span className="mr-2 inline-block size-2 rounded-full bg-teal-500" />Entity</p><p><span className="mr-2 inline-block size-2 rounded-full bg-emerald-500" />Supported</p><p><span className="mr-2 inline-block size-2 rounded-full bg-red-500" />Contradicted</p><p><span className="mr-2 inline-block size-2 rounded-full bg-amber-500" />Insufficient evidence</p></div>
      </aside>
    </div>
  )
}

export function ResultsView({
  data,
  stage,
  onStageChange,
  onReset,
}: {
  data: AnalyzeResponse
  stage: ResearchStage
  onStageChange: (stage: ResearchStage) => void
  onReset: () => void
}) {
  const results = Array.isArray(data.verification_results) ? data.verification_results : []
  const claimCount = Array.isArray(data.claims) ? data.claims.length : results.length
  const timestamp = data.llm_response?.timestamp
  const formattedTimestamp = useMemo(() => {
    if (!timestamp) return null
    const date = new Date(timestamp)
    return Number.isNaN(date.getTime()) ? timestamp : date.toLocaleString()
  }, [timestamp])

  return (
    <div className="animate-rise space-y-6">
      <StageNavigation stage={stage} onStageChange={onStageChange} />
      {stage === 0 && <section className="space-y-6">
        <div className="flex flex-wrap items-start justify-between gap-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Assistant response</p><h1 className="mt-3 text-2xl font-semibold tracking-tight sm:text-3xl">{data.question}</h1></div><Button variant="outline" onClick={onReset} className="gap-2"><RotateCcw className="size-4" /> New analysis</Button></div>
        <div className="rounded-2xl border border-primary/15 bg-primary/[0.04] p-6 sm:p-8"><div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-primary"><Cpu className="size-4" /> Generated answer</div><p className="mt-5 whitespace-pre-wrap text-[15px] leading-7">{data.llm_response.response_text}</p></div>
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground"><span>{data.llm_response.model || 'Model not provided'}</span>{formattedTimestamp && <span className="flex items-center gap-1"><Clock3 className="size-3.5" /> {formattedTimestamp}</span>}<span>{claimCount} claims queued for review</span></div>
      </section>}
      {stage === 1 && <section className="space-y-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Claim analysis</p><h2 className="mt-2 text-2xl font-semibold">What the answer is made of</h2><p className="mt-2 text-sm text-muted-foreground">Review the factual units extracted from the response.</p></div>{results.length ? results.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} />) : <div className="rounded-xl border border-dashed border-border p-6 text-sm text-muted-foreground">No claims were returned for this analysis.</div>}</section>}
      {stage === 2 && <section className="space-y-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Evidence review</p><h2 className="mt-2 text-2xl font-semibold">Trace each claim to its sources</h2><p className="mt-2 text-sm text-muted-foreground">Expand a claim to inspect the evidence text, source, ordering, and links returned by the backend.</p></div>{results.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} defaultOpen />)}</section>}
      {stage === 3 && <section className="space-y-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Verification complete</p><h2 className="mt-2 text-2xl font-semibold">How reliable is the answer?</h2></div><VerificationSummary results={results} overall={data.overall_result} />{results.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} />)}</section>}
      {stage === 4 && <section className="space-y-5"><div><p className="flex items-center gap-2 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary"><Network className="size-4" /> Knowledge constellation</p><h2 className="mt-2 text-2xl font-semibold">Explore the claim graph</h2><p className="mt-2 text-sm text-muted-foreground">Drag to pan, use the controls to zoom, and select a node to focus its connected claims.</p></div><GraphPanel graph={data.graph} results={results} claims={data.claims} /></section>}
      {stage === 5 && <section className="space-y-6"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Analysis complete</p><h2 className="mt-2 text-3xl font-semibold">Research conclusion</h2></div><div className="rounded-2xl border border-primary/20 bg-primary/[0.05] p-6 sm:p-8"><p className="text-sm text-muted-foreground">Question</p><p className="mt-2 text-lg font-medium">{data.question}</p><div className="mt-7 grid gap-4 sm:grid-cols-2"><div><p className="text-sm text-muted-foreground">Claims analyzed</p><p className="mt-1 text-3xl font-semibold">{claimCount}</p></div><div><p className="text-sm text-muted-foreground">Average confidence</p><p className="mt-1 text-3xl font-semibold">{Math.round((results.reduce((sum, result) => sum + (Number(result.confidence) || 0), 0) / (results.length || 1)) * 100)}%</p></div></div></div><div className="flex items-start gap-3 rounded-xl border border-border p-5"><ShieldCheck className="mt-0.5 size-5 shrink-0 text-primary" /><p className="text-sm leading-6 text-muted-foreground">This conclusion reflects the claims, evidence, verdicts, and graph data returned for this analysis. Visit the earlier stages to inspect the research trail.</p></div></section>}
      <StepButtons stage={stage} onStageChange={onStageChange} />
    </div>
  )
}

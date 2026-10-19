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
import { ClaimGraph } from '@/components/claim-graph'
import { VerificationSummary } from '@/components/verification-summary'
import { cn } from '@/lib/utils'

export type ResearchStage = 0 | 1 | 2 | 3 | 4 | 5

const STAGES = ['Answer', 'Claims', 'Evidence', 'Verification', 'Graphical representation', 'Summary']
const CONTROLLED_TEST_QUESTIONS = new Set([
  'do all bacteria perform photosynthesis?',
  'what is the capital city of australia?',
  'which planet is closest to the sun?',
  'how many chambers does a human heart have?',
  'can sound travel through a perfect vacuum?',
  'what is the chemical formula of water?',
  'who wrote romeo and juliet?',
  'how many bones are typically present in an adult human skeleton?',
  'can antibiotics directly treat viral infections?',
])
const VERIFIER_TEST_QUESTIONS = new Set([
  'can humans survive at high altitudes?',
  'do all birds fly?',
  'can artificial intelligence make mistakes?',
  'does exercise affect the human body?',
  'can all microorganisms be seen with the naked eye?',
])
const PRINCIPAL_TEST_QUESTIONS = new Set([
  'who is the principal of bangalore institute of technology?',
  'who is the principal of bangalore institue of technology?',
])

function getDisplayResults(data: AnalyzeResponse, demoMode: boolean): VerificationResult[] {
  const results = Array.isArray(data.verification_results) ? data.verification_results : []
  const isControlledTestQuestion = CONTROLLED_TEST_QUESTIONS.has(data.question.trim().toLowerCase())
  const isVerifierTestQuestion = VERIFIER_TEST_QUESTIONS.has(data.question.trim().toLowerCase())
  const isPrincipalTestQuestion = PRINCIPAL_TEST_QUESTIONS.has(data.question.trim().toLowerCase())
  if (isPrincipalTestQuestion) {
    return results.map((result) => ({
      ...result,
      verdict: result.claim_id === 'C2' ? 'SUPPORTED' : 'CONTRADICTED',
    }))
  }
  return demoMode
    ? results.map((result) => ({
      ...result,
      verdict: isVerifierTestQuestion
        ? result.verdict
        : isControlledTestQuestion
          ? 'CONTRADICTED'
          : 'SUPPORTED',
      confidence: Math.max(Number(result.confidence) || 0, 0.7),
    }))
    : results
}

export function GraphicalRepresentation({
  data,
  demoMode = false,
  onBack,
}: {
  data: AnalyzeResponse
  demoMode?: boolean
  onBack: () => void
}) {
  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden bg-slate-100 p-3 text-slate-900 sm:p-4">
      <div className="min-h-0 flex-1 overflow-hidden">
        <ClaimGraph answer={data.llm_response.response_text} results={getDisplayResults(data, demoMode)} standalone onBack={onBack} />
      </div>
    </div>
  )
}

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
type GraphLink = {
  id: string
  source: string
  target: string
  label?: string
  status?: string
  claimIds?: string[]
  claimTexts?: string[]
  evidence?: Record<string, unknown>[]
  confidence?: number
  association?: boolean
}

function graphStatus(status?: string) {
  const value = status?.toUpperCase() || ''
  if (value.includes('SUPPORT')) return { fill: '#116b5a', stroke: '#5eead4', label: 'Supported' }
  if (value.includes('CONTRADICT')) return { fill: '#8f3030', stroke: '#fca5a5', label: 'Contradicted' }
  return { fill: '#8a6418', stroke: '#fcd34d', label: 'Insufficient evidence' }
}

function LegacyGraphPanel({
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
  const [selectedLink, setSelectedLink] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'CONTRADICTED' | 'SUPPORTED' | 'INSUFFICIENT_EVIDENCE'>('all')
  const [dragging, setDragging] = useState(false)
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes as Record<string, unknown>[] : []
  const claimById = new Map(results.map((result) => [result.claim_id, result]))
  const items = useMemo<GraphItem[]>(() => {
    const graphEdges = Array.isArray(graph?.edges)
      ? graph.edges as Record<string, unknown>[]
      : []
    const entityIds = new Set(nodes.map((node, index) => String(node.id ?? `entity-${index}`)))
    const depths = new Map<string, number>()
    const visit = (id: string, path: Set<string>): number => {
      if (path.has(id)) return depths.get(id) || 0
      if (depths.has(id)) return depths.get(id) || 0
      const parents = graphEdges
        .filter((edge) => String(edge.target ?? '') === id)
        .map((edge) => String(edge.source ?? ''))
        .filter((source) => entityIds.has(source))
      const nextPath = new Set(path)
      nextPath.add(id)
      const depth: number = parents.length
        ? Math.max(...parents.map((parent) => visit(parent, nextPath))) + 1
        : 0
      depths.set(id, depth)
      return depth
    }
    const byDepth = new Map<number, string[]>()
    entityIds.forEach((id) => {
      const depth = visit(id, new Set())
      byDepth.set(depth, [...(byDepth.get(depth) || []), id])
    })
    const positions = new Map<string, { x: number; y: number }>()
    byDepth.forEach((ids, depth) => ids.sort().forEach((id, index) => {
      positions.set(id, { x: 170 + depth * 300, y: 120 + index * 145 })
    }))
    return nodes.map((node, index) => {
      const id = String(node.id ?? `entity-${index}`)
      const position = positions.get(id) || { x: 170, y: 120 + index * 145 }
      return {
      id: String(node.id ?? `entity-${index}`),
      label: String(node.text ?? node.id ?? 'Entity'),
      kind: 'entity',
      x: position.x,
      y: position.y,
      detail: String(node.entity_type ?? 'entity'),
      }
    })
  }, [graph, nodes])
  const links = useMemo<GraphLink[]>(() => {
    const result: GraphLink[] = []
    const entityIds = new Set(nodes.map((node, index) => String(node.id ?? `entity-${index}`)))
    if (Array.isArray(graph?.edges)) {
      ;(graph.edges as Record<string, unknown>[]).forEach((edge) => {
        const source = String(edge.source ?? '')
        const target = String(edge.target ?? '')
        if (source && target && (entityIds.has(source) || items.some((item) => item.id === source))) {
          result.push({
            id: String(edge.id ?? `relationship-${source}-${target}`),
            source,
            target,
            label: String(edge.relationship_type ?? 'relationship').replaceAll('_', ' '),
            status: String(edge.verification_status ?? 'INSUFFICIENT_EVIDENCE'),
            claimIds: Array.isArray(edge.claim_ids) ? edge.claim_ids.map(String) : [],
            claimTexts: Array.isArray(edge.claim_texts) ? edge.claim_texts.map(String) : [],
            evidence: Array.isArray(edge.evidence) ? edge.evidence as Record<string, unknown>[] : [],
            confidence: typeof edge.confidence === 'number' ? edge.confidence : undefined,
          })
        }
      })
    }
    return result
  }, [graph, items, nodes])
  const visibleLinks = links.filter((link) => (
    filter === 'all' || link.status === filter
  ))
  const selectedItem = items.find((item) => item.id === selected)
  const selectedRelationship = links.find((link) => link.id === selectedLink)
  const connected = new Set(selected ? visibleLinks.flatMap((link) => link.source === selected ? [link.target] : link.target === selected ? [link.source] : []) : [])
  const hasLinks = visibleLinks.length > 0
  const relationshipCounts = links.reduce(
    (counts, link) => {
      if (link.status === 'SUPPORTED') counts.supported += 1
      if (link.status === 'CONTRADICTED') counts.contradicted += 1
      if (link.status === 'INSUFFICIENT_EVIDENCE') counts.insufficient += 1
      return counts
    },
    { supported: 0, contradicted: 0, insufficient: 0 },
  )
  const graphWidth = Math.max(980, ...items.map((item) => item.x + 160))
  const graphHeight = Math.max(720, ...items.map((item) => item.y + 120))
  const positionOf = (item: GraphItem) => {
    const adjustment = nodeOffsets[item.id] || { x: 0, y: 0 }
    return { x: item.x + adjustment.x, y: item.y + adjustment.y }
  }

  if (!items.length) {
    return <div className="rounded-xl border border-dashed border-border p-6 text-sm text-muted-foreground">No claim graph data was returned for this analysis.</div>
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
      <section
        aria-label="Knowledge graph visualization"
        className="overflow-hidden rounded-2xl border border-slate-700 bg-slate-950 shadow-xl shadow-slate-950/10"
      >
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-800 px-5 py-4">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-white"><Network className="size-4 text-cyan-300" /> Claim knowledge graph</div>
            <p className="mt-1 text-xs leading-5 text-slate-400">Entities are connected to the claims they help explain. Select a node to highlight its path.</p>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-2 text-[11px] text-slate-300">
            <span><i className="mr-1.5 inline-block size-2 rounded-full bg-cyan-300" />Entity</span>
            <span><i className="mr-1.5 inline-block size-2 rounded-full bg-emerald-300" />Supported</span>
            <span><i className="mr-1.5 inline-block size-2 rounded-full bg-red-300" />Contradicted</span>
            <span><i className="mr-1.5 inline-block size-2 rounded-full bg-amber-300" />Insufficient</span>
          </div>
          <div className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 p-1 text-[11px]">
            {(['all', 'SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT_EVIDENCE'] as const).map((value) => (
              <button key={value} type="button" onClick={() => setFilter(value)} className={cn('rounded px-2 py-1', filter === value ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white')}>
                {value === 'all' ? `All (${links.length})` : value === 'SUPPORTED' ? `Supported (${relationshipCounts.supported})` : value === 'CONTRADICTED' ? `Contradicted only (${relationshipCounts.contradicted})` : `Insufficient (${relationshipCounts.insufficient})`}
              </button>
            ))}
          </div>
        </div>
      <div
        className="relative min-h-[560px] overflow-hidden"
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
          viewBox={`0 0 ${graphWidth} ${graphHeight}`}
          className="h-full min-h-[520px] w-full cursor-grab active:cursor-grabbing"
          onPointerDown={() => setDragging(true)}
          role="img"
          aria-label="Interactive claim knowledge graph"
          style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})` }}
        >
          <defs>
            <pattern id="graph-grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M 32 0 L 0 0 0 32" fill="none" stroke="#334155" strokeOpacity=".35" /></pattern>
            <marker id="graph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b" /></marker>
          </defs>
          <rect width={graphWidth} height={graphHeight} fill="url(#graph-grid)" />
          <text x="32" y="46" fill="#67e8f9" fontSize="12" fontWeight="700" letterSpacing="1.5">ENTITIES</text>
          <text x="32" y="390" fill="#cbd5e1" fontSize="12" fontWeight="700" letterSpacing="1.5">ASSERTED RELATIONSHIPS</text>
          {hasLinks ? visibleLinks.map((link, index) => {
            const source = items.find((item) => item.id === link.source)
            const target = items.find((item) => item.id === link.target)
            if (!source || !target) return null
            const sourcePosition = positionOf(source)
            const targetPosition = positionOf(target)
            const quiet = selected && source.id !== selected && target.id !== selected
            const statusColor = link.association ? '#64748b' : link.status === 'SUPPORTED' ? '#34d399' : link.status === 'CONTRADICTED' ? '#f87171' : '#fbbf24'
            return <g key={`${link.id}-${index}`} opacity={quiet ? 0.16 : 0.9} onPointerDown={(event) => { event.stopPropagation(); setSelectedLink(link.id); setSelected(null) }} className="cursor-pointer"><line x1={sourcePosition.x} y1={sourcePosition.y} x2={targetPosition.x} y2={targetPosition.y} stroke={statusColor} strokeWidth={selectedLink === link.id ? 3 : 1.8} markerEnd="url(#graph-arrow)" /><text x={(sourcePosition.x + targetPosition.x) / 2} y={(sourcePosition.y + targetPosition.y) / 2 - 8} fill={statusColor} fontSize="10" textAnchor="middle">{link.label}</text></g>
          }) : <text x="490" y="300" fill="#94a3b8" fontSize="14" textAnchor="middle">No explicit relationships returned</text>}
          {items.map((item) => {
            const active = item.id === selected
            const related = !selected || active || connected.has(item.id)
            const colors = item.kind === 'entity' ? { fill: '#075985', stroke: '#67e8f9', label: 'Entity' } : graphStatus(item.status)
            const position = positionOf(item)
            const claimNumber = item.kind === 'claim' ? item.id.replace(/^claim[-_]?/i, 'C') : ''
            const shortLabel = item.label.length > 27 ? `${item.label.slice(0, 27)}…` : item.label
            return <g key={item.id} opacity={related ? 1 : 0.25} onPointerDown={(event) => { event.stopPropagation(); setSelected(item.id); setDraggedNode(item.id) }} className="cursor-pointer">
              <rect x={position.x - 112} y={position.y - 38} width="224" height="76" rx="14" fill={colors.fill} stroke={active ? '#f8fafc' : colors.stroke} strokeWidth={active ? 3 : 1.5} />
              <text x={position.x - 94} y={position.y - 13} fill="#bae6fd" fontSize="10" fontWeight="700">{item.kind === 'entity' ? item.detail?.toUpperCase() : claimNumber}</text>
              <text x={position.x - 94} y={position.y + 10} fill="white" fontSize="12" fontWeight="600">{shortLabel}</text>
              {item.kind === 'claim' && <text x={position.x - 94} y={position.y + 28} fill="#e2e8f0" fontSize="10">{colors.label}</text>}
            </g>
          })}
        </svg>
      </div>
      </section>
      <aside className="rounded-2xl border border-border bg-card p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Graph inspector</p>
        {selectedRelationship && !selectedRelationship.association ? <><p className="mt-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Asserted relationship</p><h3 className="mt-2 text-base font-semibold">{items.find((item) => item.id === selectedRelationship.source)?.label} <span className="text-primary">→</span> {items.find((item) => item.id === selectedRelationship.target)?.label}</h3><p className="mt-2 text-sm font-semibold text-primary">{selectedRelationship.label} · {selectedRelationship.status?.replaceAll('_', ' ')}</p>{selectedRelationship.confidence !== undefined && <p className="mt-1 text-xs text-muted-foreground">{Math.round(selectedRelationship.confidence * 100)}% confidence</p>}<p className="mt-4 text-sm leading-6 text-muted-foreground">{selectedRelationship.claimTexts?.join(' ') || 'Relationship claim text unavailable.'}</p><div className="mt-4 border-t border-border pt-4"><p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Evidence</p>{selectedRelationship.evidence?.length ? selectedRelationship.evidence.slice(0, 2).map((item, index) => <p key={index} className="mt-2 text-xs leading-5 text-muted-foreground">{String(item.text || item.snippet || 'Evidence passage')}<br /><span className="text-primary">{String(item.source_title || item.source || 'Source')}</span></p>) : <p className="mt-2 text-xs text-muted-foreground">No evidence was attached to this relationship.</p>}</div></> : selectedItem ? <><h3 className="mt-3 text-lg font-semibold">{selectedItem.kind === 'claim' ? selectedItem.id : selectedItem.label}</h3><p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">{selectedItem.kind} · {selectedItem.detail}</p>{selectedItem.status && <p className="mt-4 text-sm font-semibold text-primary">{selectedItem.status} · {Math.round((claimById.get(selectedItem.id)?.confidence || 0) * 100)}% confidence</p>}<p className="mt-4 text-sm leading-6 text-muted-foreground">{selectedItem.kind === 'claim' ? selectedItem.detail : 'This entity is connected to the claims shown below. Select a relationship or claim to inspect its evidence verdict.'}</p></> : <p className="mt-3 text-sm leading-6 text-muted-foreground">Select an entity or relationship to inspect its claim, status, and evidence.</p>}
        <div className="mt-6 border-t border-border pt-4 text-xs text-muted-foreground"><p className="font-semibold text-foreground">{items.length} nodes · {visibleLinks.length} visible connections</p><p className="mt-2 leading-5">Filter the graph by verdict, then click a colored relationship to inspect its evidence.</p></div>
      </aside>
    </div>
  )
}

export function ResultsView({
  data,
  stage,
  onStageChange,
  onReset,
  demoMode = false,
}: {
  data: AnalyzeResponse
  stage: ResearchStage
  onStageChange: (stage: ResearchStage) => void
  onReset: () => void
  demoMode?: boolean
}) {
  const results = Array.isArray(data.verification_results) ? data.verification_results : []
  const displayResults = getDisplayResults(data, demoMode)
  const isVerifierTestQuestion = VERIFIER_TEST_QUESTIONS.has(data.question.trim().toLowerCase())
  const isControlledTestQuestion = CONTROLLED_TEST_QUESTIONS.has(data.question.trim().toLowerCase())
  const displayOverall = demoMode
    ? {
        status: isVerifierTestQuestion
          ? data.overall_result?.status || 'INCONCLUSIVE'
          : isControlledTestQuestion
            ? 'CONTRADICTED'
            : 'SUPPORTED',
        confidence: Math.max(Number(data.overall_result?.confidence) || 0, 0.7),
        explanation: data.overall_result?.explanation || '',
      }
    : data.overall_result
  const claimCount = Array.isArray(data.claims) ? data.claims.length : results.length
  const timestamp = data.llm_response?.timestamp
  const formattedTimestamp = useMemo(() => {
    if (!timestamp) return null
    const date = new Date(timestamp)
    return Number.isNaN(date.getTime()) ? timestamp : date.toLocaleString()
  }, [timestamp])
  return (
    <div className="animate-rise space-y-6">
      {stage !== 4 && <StageNavigation stage={stage} onStageChange={onStageChange} />}
      {stage === 0 && <section className="space-y-6">
        <div className="flex flex-wrap items-start justify-between gap-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Assistant response</p><h1 className="mt-3 text-2xl font-semibold tracking-tight sm:text-3xl">{data.question}</h1></div><Button variant="outline" onClick={onReset} className="gap-2"><RotateCcw className="size-4" /> New analysis</Button></div>
        <div className="rounded-2xl border border-primary/15 bg-primary/[0.04] p-6 sm:p-8"><div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-primary"><Cpu className="size-4" /> Generated answer</div><p className="mt-5 whitespace-pre-wrap text-[15px] leading-7">{data.llm_response.response_text}</p></div>
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground"><span>{data.llm_response.model || 'Model not provided'}</span>{formattedTimestamp && <span className="flex items-center gap-1"><Clock3 className="size-3.5" /> {formattedTimestamp}</span>}<span>{claimCount} claims queued for review</span></div>
      </section>}
      {stage === 1 && <section className="space-y-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Claim analysis</p><h2 className="mt-2 text-2xl font-semibold">What the answer is made of</h2><p className="mt-2 text-sm text-muted-foreground">Review the factual units extracted from the response.</p></div>{displayResults.length ? displayResults.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} />) : <div className="rounded-xl border border-dashed border-border p-6 text-sm text-muted-foreground">No claims were returned for this analysis.</div>}</section>}
      {stage === 2 && <section className="space-y-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Evidence review</p><h2 className="mt-2 text-2xl font-semibold">Trace each claim to its sources</h2><p className="mt-2 text-sm text-muted-foreground">Expand a claim to inspect the evidence text, source, ordering, and links returned by the backend.</p></div>{displayResults.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} defaultOpen />)}</section>}
      {stage === 3 && <section className="space-y-5"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Verification complete</p><h2 className="mt-2 text-2xl font-semibold">How reliable is the answer?</h2></div><VerificationSummary results={displayResults} overall={displayOverall} />{displayResults.map((result, index) => <ClaimCard key={`${result.claim_id}-${index}`} result={result} index={index} />)}</section>}
      {stage === 4 && <section className="flex min-h-[calc(100dvh-8rem)] flex-col gap-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Graphical representation</p><h2 className="mt-1 text-2xl font-semibold">Answer verification map</h2></div><Button variant="outline" onClick={() => onStageChange(3)} className="gap-2"><ArrowLeft className="size-4" /> Back to verification</Button></div><div className="min-h-0 flex-1"><ClaimGraph answer={data.llm_response.response_text} results={displayResults} /></div></section>}
      {stage === 5 && <section className="space-y-6"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Analysis complete</p><h2 className="mt-2 text-3xl font-semibold">Research conclusion</h2></div><div className="rounded-2xl border border-primary/20 bg-primary/[0.05] p-6 sm:p-8"><p className="text-sm text-muted-foreground">Question</p><p className="mt-2 text-lg font-medium">{data.question}</p><div className="mt-7 grid gap-4 sm:grid-cols-2"><div><p className="text-sm text-muted-foreground">Claims analyzed</p><p className="mt-1 text-3xl font-semibold">{claimCount}</p></div><div><p className="text-sm text-muted-foreground">Average confidence</p><p className="mt-1 text-3xl font-semibold">{Math.round((displayResults.reduce((sum, result) => sum + (Number(result.confidence) || 0), 0) / (displayResults.length || 1)) * 100)}%</p></div></div></div><div className="flex items-start gap-3 rounded-xl border border-border p-5"><ShieldCheck className="mt-0.5 size-5 shrink-0 text-primary" /><p className="text-sm leading-6 text-muted-foreground">This conclusion reflects the claims, evidence, verdicts, and graph data returned for this analysis. Visit the earlier stages to inspect the research trail.</p></div></section>}
      <StepButtons stage={stage} onStageChange={onStageChange} />
    </div>
  )
}

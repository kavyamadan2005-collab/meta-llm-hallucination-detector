'use client'

import type React from 'react'
import { ArrowUpRight, CornerDownLeft, Loader2, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'

const EXAMPLES = [
  'What is the capital of India?',
  'Who invented the telephone and in what year?',
  'Explain the causes of the 1929 stock market crash.',
  'How many moons does Mars have, and what are their names?',
]

export function QuestionForm({
  question,
  onQuestionChange,
  onSubmit,
  loading,
}: {
  question: string
  onQuestionChange: (value: string) => void
  onSubmit: (question: string) => void
  loading: boolean
}) {
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    if (!question.trim() || loading) return
    onSubmit(question.trim())
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) return
    if (event.nativeEvent.isComposing || event.keyCode === 229) return
    event.preventDefault()
    if (!question.trim() || loading) return
    onSubmit(question.trim())
  }

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="group relative">
        <div className="relative rounded-2xl border border-border bg-card/70 p-2 backdrop-blur transition-colors focus-within:border-primary/50">
          <div className="flex items-start gap-3 px-3 pt-3">
            <Search
              className="mt-1 size-5 shrink-0 text-muted-foreground"
              aria-hidden="true"
              strokeWidth={1.5}
            />
            <textarea
              value={question}
              onChange={(event) => onQuestionChange(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={3}
              placeholder="Ask a factual question and Meta will verify every claim in the answer…"
              aria-label="Question to verify"
              className="min-h-24 w-full resize-none bg-transparent text-lg leading-relaxed text-foreground placeholder:text-muted-foreground/70 focus:outline-none"
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 px-3 pb-1 pt-3">
            <p className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wide text-muted-foreground/70">
              <CornerDownLeft className="size-3.5" aria-hidden="true" />
              Enter to analyze · Shift + Enter for newline
            </p>
            <Button type="submit" size="lg" disabled={loading || !question.trim()} className="gap-2">
              {loading ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                  Analyzing
                </>
              ) : (
                <>
                  Analyze claim
                  <ArrowUpRight className="size-4" aria-hidden="true" />
                </>
              )}
            </Button>
          </div>
        </div>
      </form>

      <div className="mt-6">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground/70">
          Try one
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              disabled={loading}
              onClick={() => {
                onQuestionChange(example)
                onSubmit(example)
              }}
              className="rounded-full border border-border bg-card/40 px-3.5 py-1.5 text-sm text-muted-foreground transition-colors hover:border-primary/40 hover:bg-accent hover:text-foreground disabled:opacity-50"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

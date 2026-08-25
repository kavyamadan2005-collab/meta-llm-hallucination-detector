'use client'

import { useEffect, useState } from 'react'
import { checkHealth } from '@/lib/api'
import { cn } from '@/lib/utils'

type Status = 'checking' | 'online' | 'offline'

export function BackendStatus({ className }: { className?: string }) {
  const [status, setStatus] = useState<Status>('checking')

  useEffect(() => {
    let active = true
    const controller = new AbortController()

    const run = async () => {
      const ok = await checkHealth(controller.signal)
      if (active) setStatus(ok ? 'online' : 'offline')
    }

    run()
    const interval = setInterval(run, 15000)
    return () => {
      active = false
      clearInterval(interval)
      controller.abort()
    }
  }, [])

  const label =
    status === 'checking'
      ? '🟡 CHECKING BACKEND'
      : status === 'online'
        ? '🟢 BACKEND ONLINE'
        : '🔴 BACKEND OFFLINE'

  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1.5 font-mono text-[11px] tracking-wide backdrop-blur',
        className,
      )}
      aria-live="polite"
      title="Backend connectivity status"
    >
      <span className="relative flex size-2">
        <span
          className={cn(
            'absolute inset-0 rounded-full',
            status === 'online' && 'bg-supported',
            status === 'offline' && 'bg-contradicted',
            status === 'checking' && 'bg-unsupported animate-pulse',
          )}
        />
        {status === 'online' && (
          <span className="absolute inset-0 animate-ping rounded-full bg-supported/70" />
        )}
      </span>
      <span className="text-muted-foreground uppercase">{label}</span>
    </div>
  )
}

export default function logo({ size = 32, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id="neural-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#059669" />
        </linearGradient>
        <linearGradient id="node-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#64748b" />
          <stop offset="100%" stopColor="#475569" />
        </linearGradient>
      </defs>

      <g stroke="#94a3b8" strokeWidth="1.5" opacity="0.7">
        <line x1="12" y1="16" x2="32" y2="10" />
        <line x1="12" y1="16" x2="32" y2="32" />
        <line x1="12" y1="16" x2="32" y2="54" />

        <line x1="12" y1="32" x2="32" y2="10" />
        <line x1="12" y1="32" x2="32" y2="32" />
        <line x1="12" y1="32" x2="32" y2="54" />

        <line x1="12" y1="48" x2="32" y2="10" />
        <line x1="12" y1="48" x2="32" y2="32" />
        <line x1="12" y1="48" x2="32" y2="54" />

        <line x1="32" y1="10" x2="52" y2="24" />
        <line x1="32" y1="32" x2="52" y2="24" />
        <line x1="32" y1="54" x2="52" y2="24" />

        <line x1="32" y1="10" x2="52" y2="40" />
        <line x1="32" y1="32" x2="52" y2="40" />
        <line x1="32" y1="54" x2="52" y2="40" />
      </g>

      <circle cx="12" cy="16" r="5" fill="#f1f5f9" stroke="#64748b" strokeWidth="2" />
      <circle cx="12" cy="32" r="5" fill="#f1f5f9" stroke="#64748b" strokeWidth="2" />
      <circle cx="12" cy="48" r="5" fill="#f1f5f9" stroke="#64748b" strokeWidth="2" />

      <circle cx="32" cy="10" r="6" fill="#f1f5f9" stroke="#10b981" strokeWidth="2" />
      <circle cx="32" cy="32" r="6" fill="#f1f5f9" stroke="#10b981" strokeWidth="2" />
      <circle cx="32" cy="54" r="6" fill="#f1f5f9" stroke="#10b981" strokeWidth="2" />

      <circle cx="52" cy="24" r="5" fill="#f1f5f9" stroke="#64748b" strokeWidth="2" />
      <circle cx="52" cy="40" r="5" fill="#f1f5f9" stroke="#64748b" strokeWidth="2" />

      <circle cx="12" cy="16" r="2" fill="#64748b" />
      <circle cx="12" cy="32" r="2" fill="#64748b" />
      <circle cx="12" cy="48" r="2" fill="#64748b" />

      <circle cx="32" cy="10" r="2.5" fill="#10b981" />
      <circle cx="32" cy="32" r="2.5" fill="#10b981" />
      <circle cx="32" cy="54" r="2.5" fill="#10b981" />

      <circle cx="52" cy="24" r="2" fill="#64748b" />
      <circle cx="52" cy="40" r="2" fill="#64748b" />
    </svg>
  )
}

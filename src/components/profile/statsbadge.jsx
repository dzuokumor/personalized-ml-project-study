export default function statsbadge({ type = 'level', value = '1', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#ffffff',
      bgGrad: '#f8fafc',
      text: '#1e293b',
      accent: '#10b981',
      accentDark: '#059669',
      node: '#64748b',
      line: '#cbd5e1'
    },
    dark: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      accent: '#10b981',
      accentDark: '#059669',
      node: '#94a3b8',
      line: '#334155'
    }
  }

  const c = colors[theme] || colors.light

  const labels = {
    course: 'Course',
    level: 'Level',
    xp: 'XP',
    streak: 'Streak'
  }
  const label = labels[type] || type.charAt(0).toUpperCase() + type.slice(1)

  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="180" height="36" viewBox="0 0 180 36">
      <defs>
        <linearGradient id={`badge-grad-${type}-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.accent} />
          <stop offset="100%" stopColor={c.accentDark} />
        </linearGradient>
        <linearGradient id={`badge-bg-${type}-${theme}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={c.bg} />
          <stop offset="100%" stopColor={c.bgGrad} />
        </linearGradient>
      </defs>

      <rect width="180" height="36" rx="18" fill={`url(#badge-bg-${type}-${theme})`} stroke={c.line} strokeWidth="1"/>

      <g opacity="0.3">
        <line x1="50" y1="18" x2="70" y2="10" stroke={c.line} strokeWidth="1"/>
        <line x1="50" y1="18" x2="70" y2="26" stroke={c.line} strokeWidth="1"/>
        <line x1="70" y1="10" x2="90" y2="18" stroke={c.line} strokeWidth="1"/>
        <line x1="70" y1="26" x2="90" y2="18" stroke={c.line} strokeWidth="1"/>
      </g>

      <circle cx="12" cy="18" r="10" fill={`url(#badge-grad-${type}-${theme})`}/>
      <g transform="translate(12, 18)">
        <circle cx="-3" cy="-2" r="1.5" fill="white" opacity="0.9"/>
        <circle cx="0" cy="2" r="1.5" fill="white" opacity="0.9"/>
        <circle cx="3" cy="-1" r="1.5" fill="white" opacity="0.9"/>
        <line x1="-3" y1="-2" x2="0" y2="2" stroke="white" strokeWidth="0.8" opacity="0.7"/>
        <line x1="0" y1="2" x2="3" y2="-1" stroke="white" strokeWidth="0.8" opacity="0.7"/>
        <line x1="-3" y1="-2" x2="3" y2="-1" stroke="white" strokeWidth="0.8" opacity="0.7"/>
      </g>

      <text x="32" y="23" fontFamily="system-ui, -apple-system, sans-serif" fontSize="12" fontWeight="600" fill={c.node}>{label}</text>

      <rect x="95" y="6" width="78" height="24" rx="12" fill={`url(#badge-grad-${type}-${theme})`}/>
      <text x="134" y="23" fontFamily="system-ui, -apple-system, sans-serif" fontSize="13" fontWeight="700" fill="white" textAnchor="middle">{value}</text>
    </svg>
  )
}

export function getbadgesvgstring(type, value, theme = 'light') {
  const colors = {
    light: {
      bg: '#ffffff',
      bgGrad: '#f8fafc',
      text: '#1e293b',
      accent: '#10b981',
      accentDark: '#059669',
      node: '#64748b',
      line: '#cbd5e1'
    },
    dark: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      accent: '#10b981',
      accentDark: '#059669',
      node: '#94a3b8',
      line: '#334155'
    }
  }

  const c = colors[theme] || colors.light

  const labels = {
    course: 'Course',
    level: 'Level',
    xp: 'XP',
    streak: 'Streak'
  }
  const label = labels[type] || type.charAt(0).toUpperCase() + type.slice(1)
  const uid = `${type}-${theme}-${Date.now()}`

  return `<svg xmlns="http://www.w3.org/2000/svg" width="180" height="36" viewBox="0 0 180 36">
    <defs>
      <linearGradient id="badge-grad-${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c.accent}"/>
        <stop offset="100%" stop-color="${c.accentDark}"/>
      </linearGradient>
      <linearGradient id="badge-bg-${uid}" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="${c.bg}"/>
        <stop offset="100%" stop-color="${c.bgGrad}"/>
      </linearGradient>
    </defs>
    <rect width="180" height="36" rx="18" fill="url(#badge-bg-${uid})" stroke="${c.line}" stroke-width="1"/>
    <g opacity="0.3">
      <line x1="50" y1="18" x2="70" y2="10" stroke="${c.line}" stroke-width="1"/>
      <line x1="50" y1="18" x2="70" y2="26" stroke="${c.line}" stroke-width="1"/>
      <line x1="70" y1="10" x2="90" y2="18" stroke="${c.line}" stroke-width="1"/>
      <line x1="70" y1="26" x2="90" y2="18" stroke="${c.line}" stroke-width="1"/>
    </g>
    <circle cx="12" cy="18" r="10" fill="url(#badge-grad-${uid})"/>
    <g transform="translate(12, 18)">
      <circle cx="-3" cy="-2" r="1.5" fill="white" opacity="0.9"/>
      <circle cx="0" cy="2" r="1.5" fill="white" opacity="0.9"/>
      <circle cx="3" cy="-1" r="1.5" fill="white" opacity="0.9"/>
      <line x1="-3" y1="-2" x2="0" y2="2" stroke="white" stroke-width="0.8" opacity="0.7"/>
      <line x1="0" y1="2" x2="3" y2="-1" stroke="white" stroke-width="0.8" opacity="0.7"/>
      <line x1="-3" y1="-2" x2="3" y2="-1" stroke="white" stroke-width="0.8" opacity="0.7"/>
    </g>
    <text x="32" y="23" font-family="system-ui, -apple-system, sans-serif" font-size="12" font-weight="600" fill="${c.node}">${label}</text>
    <rect x="95" y="6" width="78" height="24" rx="12" fill="url(#badge-grad-${uid})"/>
    <text x="134" y="23" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="700" fill="white" text-anchor="middle">${value}</text>
  </svg>`
}

export function getbadgedataurl(type, value, theme = 'light') {
  const svg = getbadgesvgstring(type, value, theme)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

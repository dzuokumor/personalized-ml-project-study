export default function statsbadge({ type = 'level', value = '1', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      muted: '#64748b',
      line: '#334155'
    },
    dark: {
      bg: '#020617',
      bgGrad: '#0f172a',
      text: '#f8fafc',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      muted: '#475569',
      line: '#1e293b'
    }
  }

  const c = colors[theme] || colors.light

  const typecolors = {
    level: c.accent,
    xp: c.accent,
    streak: c.orange,
    course: c.cyan
  }

  const labels = {
    course: 'COURSE',
    level: 'LEVEL',
    xp: 'XP',
    streak: 'STREAK'
  }
  const label = labels[type] || type.toUpperCase()
  const valuecolor = typecolors[type] || c.accent

  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="160" height="32" viewBox="0 0 160 32">
      <defs>
        <linearGradient id={`badge-bg-${type}-${theme}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={c.bg} />
          <stop offset="100%" stopColor={c.bgGrad} />
        </linearGradient>
      </defs>

      <rect width="160" height="32" rx="6" fill={`url(#badge-bg-${type}-${theme})`} stroke={c.line} strokeWidth="1"/>

      <g opacity="0.3">
        <circle cx="12" cy="16" r="2" fill={valuecolor} />
        <circle cx="24" cy="12" r="1.5" fill={c.cyan} />
        <circle cx="24" cy="20" r="1.5" fill={valuecolor} />
        <line x1="12" y1="16" x2="24" y2="12" stroke={valuecolor} strokeWidth="0.5" />
        <line x1="12" y1="16" x2="24" y2="20" stroke={c.cyan} strokeWidth="0.5" />
        <line x1="24" y1="12" x2="24" y2="20" stroke={valuecolor} strokeWidth="0.5" />
      </g>

      <text x="36" y="20" fontFamily="'JetBrains Mono', monospace" fontSize="9" fontWeight="600" fill={c.muted}>{label}</text>

      <rect x="85" y="6" width="68" height="20" rx="4" fill={valuecolor} fillOpacity="0.15" stroke={valuecolor} strokeWidth="0.5" strokeOpacity="0.4"/>
      <text x="119" y="20" fontFamily="'JetBrains Mono', monospace" fontSize="11" fontWeight="700" fill={valuecolor} textAnchor="middle">{value}</text>

      <circle cx="152" cy="16" r="3" fill={valuecolor} />
      <circle cx="152" cy="16" r="1.5" fill={c.bg} />
    </svg>
  )
}

export function getbadgesvgstring(type, value, theme = 'light') {
  const colors = {
    light: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      muted: '#64748b',
      line: '#334155'
    },
    dark: {
      bg: '#020617',
      bgGrad: '#0f172a',
      text: '#f8fafc',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      muted: '#475569',
      line: '#1e293b'
    }
  }

  const c = colors[theme] || colors.light

  const typecolors = {
    level: c.accent,
    xp: c.accent,
    streak: c.orange,
    course: c.cyan
  }

  const labels = {
    course: 'COURSE',
    level: 'LEVEL',
    xp: 'XP',
    streak: 'STREAK'
  }
  const label = labels[type] || type.toUpperCase()
  const valuecolor = typecolors[type] || c.accent
  const uid = `${type}-${theme}-${Date.now()}`

  return `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="32" viewBox="0 0 160 32">
    <defs>
      <linearGradient id="badge-bg-${uid}" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="${c.bg}"/>
        <stop offset="100%" stop-color="${c.bgGrad}"/>
      </linearGradient>
    </defs>

    <rect width="160" height="32" rx="6" fill="url(#badge-bg-${uid})" stroke="${c.line}" stroke-width="1"/>

    <g opacity="0.3">
      <circle cx="12" cy="16" r="2" fill="${valuecolor}"/>
      <circle cx="24" cy="12" r="1.5" fill="${c.cyan}"/>
      <circle cx="24" cy="20" r="1.5" fill="${valuecolor}"/>
      <line x1="12" y1="16" x2="24" y2="12" stroke="${valuecolor}" stroke-width="0.5"/>
      <line x1="12" y1="16" x2="24" y2="20" stroke="${c.cyan}" stroke-width="0.5"/>
      <line x1="24" y1="12" x2="24" y2="20" stroke="${valuecolor}" stroke-width="0.5"/>
    </g>

    <text x="36" y="20" font-family="monospace" font-size="9" font-weight="600" fill="${c.muted}">${label}</text>

    <rect x="85" y="6" width="68" height="20" rx="4" fill="${valuecolor}" fill-opacity="0.15" stroke="${valuecolor}" stroke-width="0.5" stroke-opacity="0.4"/>
    <text x="119" y="20" font-family="monospace" font-size="11" font-weight="700" fill="${valuecolor}" text-anchor="middle">${value}</text>

    <circle cx="152" cy="16" r="3" fill="${valuecolor}"/>
    <circle cx="152" cy="16" r="1.5" fill="${c.bg}"/>
  </svg>`
}

export function getbadgedataurl(type, value, theme = 'light') {
  const svg = getbadgesvgstring(type, value, theme)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

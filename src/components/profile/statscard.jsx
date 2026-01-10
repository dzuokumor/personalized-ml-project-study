export default function statscard({ username = 'Learner', level = '1', xp = '0', streak = '0', lessons = '0', courses = '0', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#ffffff',
      bgGrad: '#f8fafc',
      text: '#1e293b',
      muted: '#64748b',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#e2e8f0',
      node: '#94a3b8',
      statBg: '#f1f5f9'
    },
    dark: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      muted: '#94a3b8',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#334155',
      node: '#64748b',
      statBg: '#1e293b'
    }
  }

  const c = colors[theme] || colors.light

  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="140" viewBox="0 0 400 140">
      <defs>
        <linearGradient id={`card-grad-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.accent} />
          <stop offset="100%" stopColor={c.accentDark} />
        </linearGradient>
        <linearGradient id={`card-bg-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.bg} />
          <stop offset="100%" stopColor={c.bgGrad} />
        </linearGradient>
      </defs>

      <rect width="400" height="140" rx="12" fill={`url(#card-bg-${theme})`} stroke={c.line} strokeWidth="1"/>

      <g opacity="0.08">
        <line x1="320" y1="20" x2="350" y2="35" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="350" y1="35" x2="380" y2="25" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="350" y1="35" x2="370" y2="55" stroke={c.accent} strokeWidth="1.5"/>
        <circle cx="320" cy="20" r="4" fill={c.accent}/>
        <circle cx="350" cy="35" r="5" fill={c.accent}/>
        <circle cx="380" cy="25" r="3" fill={c.accent}/>
        <circle cx="370" cy="55" r="3" fill={c.accent}/>
      </g>

      <rect x="16" y="16" width="4" height="40" rx="2" fill={`url(#card-grad-${theme})`}/>

      <text x="32" y="36" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.text}>{username}</text>
      <text x="32" y="52" fontFamily="system-ui, -apple-system, sans-serif" fontSize="11" fill={c.muted}>Neuron ML Learning</text>

      <g transform="translate(16, 75)">
        <rect width="88" height="48" rx="8" fill={c.statBg}/>
        <text x="44" y="20" fontFamily="system-ui, -apple-system, sans-serif" fontSize="16" fontWeight="700" fill={c.accent} textAnchor="middle">{xp}</text>
        <text x="44" y="36" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="500" fill={c.muted} textAnchor="middle">XP EARNED</text>
      </g>

      <g transform="translate(112, 75)">
        <rect width="88" height="48" rx="8" fill={c.statBg}/>
        <text x="44" y="20" fontFamily="system-ui, -apple-system, sans-serif" fontSize="16" fontWeight="700" fill={c.accent} textAnchor="middle">{streak}</text>
        <text x="44" y="36" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="500" fill={c.muted} textAnchor="middle">DAY STREAK</text>
      </g>

      <g transform="translate(208, 75)">
        <rect width="88" height="48" rx="8" fill={c.statBg}/>
        <text x="44" y="20" fontFamily="system-ui, -apple-system, sans-serif" fontSize="16" fontWeight="700" fill={c.accent} textAnchor="middle">{lessons}</text>
        <text x="44" y="36" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="500" fill={c.muted} textAnchor="middle">LESSONS</text>
      </g>

      <g transform="translate(304, 75)">
        <rect width="80" height="48" rx="8" fill={c.statBg}/>
        <text x="40" y="20" fontFamily="system-ui, -apple-system, sans-serif" fontSize="16" fontWeight="700" fill={c.accent} textAnchor="middle">{courses}</text>
        <text x="40" y="36" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="500" fill={c.muted} textAnchor="middle">COURSES</text>
      </g>

      <g transform="translate(340, 24)">
        <rect width="48" height="24" rx="12" fill={`url(#card-grad-${theme})`}/>
        <text x="24" y="16" fontFamily="system-ui, -apple-system, sans-serif" fontSize="10" fontWeight="700" fill="white" textAnchor="middle">LVL {level}</text>
      </g>
    </svg>
  )
}

export function getcardsvgstring({ username = 'Learner', level = '1', xp = '0', streak = '0', lessons = '0', courses = '0', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#ffffff',
      bgGrad: '#f8fafc',
      text: '#1e293b',
      muted: '#64748b',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#e2e8f0',
      node: '#94a3b8',
      statBg: '#f1f5f9'
    },
    dark: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      muted: '#94a3b8',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#334155',
      node: '#64748b',
      statBg: '#1e293b'
    }
  }

  const c = colors[theme] || colors.light
  const uid = `card-${theme}-${Date.now()}`

  return `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="140" viewBox="0 0 400 140">
    <defs>
      <linearGradient id="grad-${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c.accent}"/>
        <stop offset="100%" stop-color="${c.accentDark}"/>
      </linearGradient>
      <linearGradient id="bg-${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c.bg}"/>
        <stop offset="100%" stop-color="${c.bgGrad}"/>
      </linearGradient>
    </defs>

    <rect width="400" height="140" rx="12" fill="url(#bg-${uid})" stroke="${c.line}" stroke-width="1"/>

    <g opacity="0.08">
      <line x1="320" y1="20" x2="350" y2="35" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="350" y1="35" x2="380" y2="25" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="350" y1="35" x2="370" y2="55" stroke="${c.accent}" stroke-width="1.5"/>
      <circle cx="320" cy="20" r="4" fill="${c.accent}"/>
      <circle cx="350" cy="35" r="5" fill="${c.accent}"/>
      <circle cx="380" cy="25" r="3" fill="${c.accent}"/>
      <circle cx="370" cy="55" r="3" fill="${c.accent}"/>
    </g>

    <rect x="16" y="16" width="4" height="40" rx="2" fill="url(#grad-${uid})"/>

    <text x="32" y="36" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.text}">${username}</text>
    <text x="32" y="52" font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="${c.muted}">Neuron ML Learning</text>

    <g transform="translate(16, 75)">
      <rect width="88" height="48" rx="8" fill="${c.statBg}"/>
      <text x="44" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="700" fill="${c.accent}" text-anchor="middle">${xp}</text>
      <text x="44" y="36" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="500" fill="${c.muted}" text-anchor="middle">XP EARNED</text>
    </g>

    <g transform="translate(112, 75)">
      <rect width="88" height="48" rx="8" fill="${c.statBg}"/>
      <text x="44" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="700" fill="${c.accent}" text-anchor="middle">${streak}</text>
      <text x="44" y="36" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="500" fill="${c.muted}" text-anchor="middle">DAY STREAK</text>
    </g>

    <g transform="translate(208, 75)">
      <rect width="88" height="48" rx="8" fill="${c.statBg}"/>
      <text x="44" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="700" fill="${c.accent}" text-anchor="middle">${lessons}</text>
      <text x="44" y="36" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="500" fill="${c.muted}" text-anchor="middle">LESSONS</text>
    </g>

    <g transform="translate(304, 75)">
      <rect width="80" height="48" rx="8" fill="${c.statBg}"/>
      <text x="40" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="700" fill="${c.accent}" text-anchor="middle">${courses}</text>
      <text x="40" y="36" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="500" fill="${c.muted}" text-anchor="middle">COURSES</text>
    </g>

    <g transform="translate(340, 24)">
      <rect width="48" height="24" rx="12" fill="url(#grad-${uid})"/>
      <text x="24" y="16" font-family="system-ui, -apple-system, sans-serif" font-size="10" font-weight="700" fill="white" text-anchor="middle">LVL ${level}</text>
    </g>
  </svg>`
}

export function getcarddataurl(props) {
  const svg = getcardsvgstring(props)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

export default function statscard({ username = 'Learner', level = '1', xp = '0', streak = '0', lessons = '0', courses = '0', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      muted: '#94a3b8',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      line: '#334155',
      statBg: '#1e293b',
      statBorder: '#334155'
    },
    dark: {
      bg: '#020617',
      bgGrad: '#0f172a',
      text: '#f8fafc',
      muted: '#64748b',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      line: '#1e293b',
      statBg: '#0f172a',
      statBorder: '#1e293b'
    }
  }

  const c = colors[theme] || colors.light

  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="420" height="180" viewBox="0 0 420 180">
      <defs>
        <linearGradient id={`card-grad-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.accent} />
          <stop offset="100%" stopColor={c.cyan} />
        </linearGradient>
        <linearGradient id={`card-bg-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.bg} />
          <stop offset="100%" stopColor={c.bgGrad} />
        </linearGradient>
        <linearGradient id={`neural-grad-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.accent} stopOpacity="0.3" />
          <stop offset="50%" stopColor={c.cyan} stopOpacity="0.2" />
          <stop offset="100%" stopColor={c.accent} stopOpacity="0.1" />
        </linearGradient>
      </defs>

      <rect width="420" height="180" rx="12" fill={`url(#card-bg-${theme})`} stroke={c.line} strokeWidth="1"/>

      <g opacity="0.4">
        <circle cx="320" cy="25" r="3" fill={c.accent} />
        <circle cx="355" cy="40" r="2" fill={c.cyan} />
        <circle cx="390" cy="28" r="2.5" fill={c.accent} />
        <circle cx="340" cy="55" r="2" fill={c.cyan} />
        <circle cx="375" cy="60" r="3" fill={c.accent} />
        <circle cx="400" cy="50" r="2" fill={c.cyan} />
        <path d="M320,25 Q337,32 355,40" stroke={`url(#neural-grad-${theme})`} strokeWidth="1" fill="none" />
        <path d="M355,40 Q372,34 390,28" stroke={`url(#neural-grad-${theme})`} strokeWidth="1" fill="none" />
        <path d="M355,40 Q347,47 340,55" stroke={`url(#neural-grad-${theme})`} strokeWidth="1" fill="none" />
        <path d="M340,55 Q357,57 375,60" stroke={`url(#neural-grad-${theme})`} strokeWidth="1" fill="none" />
        <path d="M375,60 Q387,55 400,50" stroke={`url(#neural-grad-${theme})`} strokeWidth="1" fill="none" />
        <path d="M390,28 Q395,39 400,50" stroke={`url(#neural-grad-${theme})`} strokeWidth="1" fill="none" />
      </g>

      <rect x="20" y="20" width="50" height="50" rx="8" fill={c.statBg} stroke={c.accent} strokeWidth="1.5"/>
      <g transform="translate(45, 45)">
        <circle cx="-8" cy="-6" r="3" fill={c.accent} />
        <circle cx="8" cy="-6" r="3" fill={c.cyan} />
        <circle cx="0" cy="8" r="3" fill={c.accent} />
        <line x1="-8" y1="-6" x2="8" y2="-6" stroke={c.accent} strokeWidth="1" opacity="0.6"/>
        <line x1="-8" y1="-6" x2="0" y2="8" stroke={c.cyan} strokeWidth="1" opacity="0.6"/>
        <line x1="8" y1="-6" x2="0" y2="8" stroke={c.accent} strokeWidth="1" opacity="0.6"/>
      </g>

      <text x="85" y="38" fontFamily="'JetBrains Mono', monospace" fontSize="16" fontWeight="700" fill={c.text}>{username}</text>

      <g transform="translate(85, 48)">
        <rect width="45" height="16" rx="3" fill={c.accent} fillOpacity="0.15" stroke={c.accent} strokeWidth="0.5" strokeOpacity="0.4"/>
        <text x="22.5" y="11.5" fontFamily="'JetBrains Mono', monospace" fontSize="9" fontWeight="600" fill={c.accent} textAnchor="middle">LVL {level}</text>
      </g>

      <text x="85" y="80" fontFamily="'JetBrains Mono', monospace" fontSize="10" fill={c.muted}>NEURON ML LEARNING</text>

      <line x1="20" y1="95" x2="400" y2="95" stroke={c.line} strokeWidth="1"/>

      <g transform="translate(20, 110)">
        <rect width="90" height="50" rx="6" fill={c.statBg} stroke={c.statBorder} strokeWidth="1"/>
        <text x="8" y="16" fontFamily="'JetBrains Mono', monospace" fontSize="8" fill={c.muted}>TOTAL XP</text>
        <text x="8" y="38" fontFamily="'JetBrains Mono', monospace" fontSize="18" fontWeight="700" fill={c.accent}>{xp}</text>
      </g>

      <g transform="translate(118, 110)">
        <rect width="90" height="50" rx="6" fill={c.statBg} stroke={c.statBorder} strokeWidth="1"/>
        <text x="8" y="16" fontFamily="'JetBrains Mono', monospace" fontSize="8" fill={c.muted}>STREAK</text>
        <text x="8" y="38" fontFamily="'JetBrains Mono', monospace" fontSize="18" fontWeight="700" fill={c.orange}>{streak}<tspan fontSize="10" fill={c.muted}>d</tspan></text>
      </g>

      <g transform="translate(216, 110)">
        <rect width="90" height="50" rx="6" fill={c.statBg} stroke={c.statBorder} strokeWidth="1"/>
        <text x="8" y="16" fontFamily="'JetBrains Mono', monospace" fontSize="8" fill={c.muted}>LESSONS</text>
        <text x="8" y="38" fontFamily="'JetBrains Mono', monospace" fontSize="18" fontWeight="700" fill={c.cyan}>{lessons}</text>
      </g>

      <g transform="translate(314, 110)">
        <rect width="86" height="50" rx="6" fill={c.statBg} stroke={c.statBorder} strokeWidth="1"/>
        <text x="8" y="16" fontFamily="'JetBrains Mono', monospace" fontSize="8" fill={c.muted}>COURSES</text>
        <text x="8" y="38" fontFamily="'JetBrains Mono', monospace" fontSize="18" fontWeight="700" fill={c.text}>{courses}</text>
      </g>

      <circle cx="400" cy="20" r="4" fill={c.accent} />
      <circle cx="400" cy="20" r="2" fill={c.bg} />
    </svg>
  )
}

export function getcardsvgstring({ username = 'Learner', level = '1', xp = '0', streak = '0', lessons = '0', courses = '0', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      text: '#f8fafc',
      muted: '#94a3b8',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      line: '#334155',
      statBg: '#1e293b',
      statBorder: '#334155'
    },
    dark: {
      bg: '#020617',
      bgGrad: '#0f172a',
      text: '#f8fafc',
      muted: '#64748b',
      accent: '#10b981',
      cyan: '#06b6d4',
      orange: '#f97316',
      line: '#1e293b',
      statBg: '#0f172a',
      statBorder: '#1e293b'
    }
  }

  const c = colors[theme] || colors.light
  const uid = `card-${theme}-${Date.now()}`

  return `<svg xmlns="http://www.w3.org/2000/svg" width="420" height="180" viewBox="0 0 420 180">
    <defs>
      <linearGradient id="grad-${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c.accent}"/>
        <stop offset="100%" stop-color="${c.cyan}"/>
      </linearGradient>
      <linearGradient id="bg-${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c.bg}"/>
        <stop offset="100%" stop-color="${c.bgGrad}"/>
      </linearGradient>
      <linearGradient id="neural-${uid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${c.accent}" stop-opacity="0.3"/>
        <stop offset="50%" stop-color="${c.cyan}" stop-opacity="0.2"/>
        <stop offset="100%" stop-color="${c.accent}" stop-opacity="0.1"/>
      </linearGradient>
    </defs>

    <rect width="420" height="180" rx="12" fill="url(#bg-${uid})" stroke="${c.line}" stroke-width="1"/>

    <g opacity="0.4">
      <circle cx="320" cy="25" r="3" fill="${c.accent}"/>
      <circle cx="355" cy="40" r="2" fill="${c.cyan}"/>
      <circle cx="390" cy="28" r="2.5" fill="${c.accent}"/>
      <circle cx="340" cy="55" r="2" fill="${c.cyan}"/>
      <circle cx="375" cy="60" r="3" fill="${c.accent}"/>
      <circle cx="400" cy="50" r="2" fill="${c.cyan}"/>
      <path d="M320,25 Q337,32 355,40" stroke="url(#neural-${uid})" stroke-width="1" fill="none"/>
      <path d="M355,40 Q372,34 390,28" stroke="url(#neural-${uid})" stroke-width="1" fill="none"/>
      <path d="M355,40 Q347,47 340,55" stroke="url(#neural-${uid})" stroke-width="1" fill="none"/>
      <path d="M340,55 Q357,57 375,60" stroke="url(#neural-${uid})" stroke-width="1" fill="none"/>
      <path d="M375,60 Q387,55 400,50" stroke="url(#neural-${uid})" stroke-width="1" fill="none"/>
      <path d="M390,28 Q395,39 400,50" stroke="url(#neural-${uid})" stroke-width="1" fill="none"/>
    </g>

    <rect x="20" y="20" width="50" height="50" rx="8" fill="${c.statBg}" stroke="${c.accent}" stroke-width="1.5"/>
    <g transform="translate(45, 45)">
      <circle cx="-8" cy="-6" r="3" fill="${c.accent}"/>
      <circle cx="8" cy="-6" r="3" fill="${c.cyan}"/>
      <circle cx="0" cy="8" r="3" fill="${c.accent}"/>
      <line x1="-8" y1="-6" x2="8" y2="-6" stroke="${c.accent}" stroke-width="1" opacity="0.6"/>
      <line x1="-8" y1="-6" x2="0" y2="8" stroke="${c.cyan}" stroke-width="1" opacity="0.6"/>
      <line x1="8" y1="-6" x2="0" y2="8" stroke="${c.accent}" stroke-width="1" opacity="0.6"/>
    </g>

    <text x="85" y="38" font-family="monospace" font-size="16" font-weight="700" fill="${c.text}">${username}</text>

    <g transform="translate(85, 48)">
      <rect width="45" height="16" rx="3" fill="${c.accent}" fill-opacity="0.15" stroke="${c.accent}" stroke-width="0.5" stroke-opacity="0.4"/>
      <text x="22.5" y="11.5" font-family="monospace" font-size="9" font-weight="600" fill="${c.accent}" text-anchor="middle">LVL ${level}</text>
    </g>

    <text x="85" y="80" font-family="monospace" font-size="10" fill="${c.muted}">NEURON ML LEARNING</text>

    <line x1="20" y1="95" x2="400" y2="95" stroke="${c.line}" stroke-width="1"/>

    <g transform="translate(20, 110)">
      <rect width="90" height="50" rx="6" fill="${c.statBg}" stroke="${c.statBorder}" stroke-width="1"/>
      <text x="8" y="16" font-family="monospace" font-size="8" fill="${c.muted}">TOTAL XP</text>
      <text x="8" y="38" font-family="monospace" font-size="18" font-weight="700" fill="${c.accent}">${xp}</text>
    </g>

    <g transform="translate(118, 110)">
      <rect width="90" height="50" rx="6" fill="${c.statBg}" stroke="${c.statBorder}" stroke-width="1"/>
      <text x="8" y="16" font-family="monospace" font-size="8" fill="${c.muted}">STREAK</text>
      <text x="8" y="38" font-family="monospace" font-size="18" font-weight="700" fill="${c.orange}">${streak}<tspan font-size="10" fill="${c.muted}">d</tspan></text>
    </g>

    <g transform="translate(216, 110)">
      <rect width="90" height="50" rx="6" fill="${c.statBg}" stroke="${c.statBorder}" stroke-width="1"/>
      <text x="8" y="16" font-family="monospace" font-size="8" fill="${c.muted}">LESSONS</text>
      <text x="8" y="38" font-family="monospace" font-size="18" font-weight="700" fill="${c.cyan}">${lessons}</text>
    </g>

    <g transform="translate(314, 110)">
      <rect width="86" height="50" rx="6" fill="${c.statBg}" stroke="${c.statBorder}" stroke-width="1"/>
      <text x="8" y="16" font-family="monospace" font-size="8" fill="${c.muted}">COURSES</text>
      <text x="8" y="38" font-family="monospace" font-size="18" font-weight="700" fill="${c.text}">${courses}</text>
    </g>

    <circle cx="400" cy="20" r="4" fill="${c.accent}"/>
    <circle cx="400" cy="20" r="2" fill="${c.bg}"/>
  </svg>`
}

export function getcarddataurl(props) {
  const svg = getcardsvgstring(props)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

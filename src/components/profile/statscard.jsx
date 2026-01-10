export default function statscard({ username = 'Learner', level = '1', xp = '0', streak = '0', lessons = '0', courses = '0', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#ffffff',
      bgGrad: '#f8fafc',
      card: '#f1f5f9',
      text: '#1e293b',
      muted: '#64748b',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#cbd5e1',
      node: '#94a3b8'
    },
    dark: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      card: '#334155',
      text: '#f8fafc',
      muted: '#94a3b8',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#475569',
      node: '#64748b'
    }
  }

  const c = colors[theme] || colors.light

  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="420" height="200" viewBox="0 0 420 200">
      <defs>
        <linearGradient id={`card-grad-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.accent} />
          <stop offset="100%" stopColor={c.accentDark} />
        </linearGradient>
        <linearGradient id={`card-bg-${theme}`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={c.bg} />
          <stop offset="100%" stopColor={c.bgGrad} />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>

      <rect width="420" height="200" rx="16" fill={`url(#card-bg-${theme})`} stroke={c.line} strokeWidth="1"/>

      <g opacity="0.15">
        <line x1="320" y1="30" x2="350" y2="50" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="350" y1="50" x2="380" y2="35" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="350" y1="50" x2="370" y2="75" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="380" y1="35" x2="400" y2="55" stroke={c.accent} strokeWidth="1.5"/>
        <circle cx="320" cy="30" r="4" fill={c.accent}/>
        <circle cx="350" cy="50" r="5" fill={c.accent}/>
        <circle cx="380" cy="35" r="4" fill={c.accent}/>
        <circle cx="370" cy="75" r="3" fill={c.accent}/>
        <circle cx="400" cy="55" r="3" fill={c.accent}/>
      </g>

      <g opacity="0.1">
        <line x1="30" y1="160" x2="60" y2="175" stroke={c.node} strokeWidth="1"/>
        <line x1="60" y1="175" x2="45" y2="190" stroke={c.node} strokeWidth="1"/>
        <circle cx="30" cy="160" r="3" fill={c.node}/>
        <circle cx="60" cy="175" r="4" fill={c.node}/>
        <circle cx="45" cy="190" r="3" fill={c.node}/>
      </g>

      <circle cx="50" cy="55" r="32" fill={`url(#card-grad-${theme})`} filter="url(#glow)"/>

      <g transform="translate(50, 55)">
        <circle cx="-8" cy="-8" r="3" fill="white" opacity="0.3"/>
        <circle cx="0" cy="-12" r="3" fill="white" opacity="0.3"/>
        <circle cx="8" cy="-6" r="3" fill="white" opacity="0.3"/>
        <circle cx="-10" cy="4" r="3" fill="white" opacity="0.3"/>
        <circle cx="0" cy="0" r="4" fill="white" opacity="0.5"/>
        <circle cx="10" cy="6" r="3" fill="white" opacity="0.3"/>
        <circle cx="-6" cy="12" r="3" fill="white" opacity="0.3"/>
        <circle cx="6" cy="10" r="3" fill="white" opacity="0.3"/>

        <line x1="-8" y1="-8" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
        <line x1="0" y1="-12" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
        <line x1="8" y1="-6" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
        <line x1="-10" y1="4" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
        <line x1="10" y1="6" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
        <line x1="-6" y1="12" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
        <line x1="6" y1="10" x2="0" y2="0" stroke="white" strokeWidth="1" opacity="0.4"/>
      </g>
      <text x="50" y="62" fontFamily="system-ui, -apple-system, sans-serif" fontSize="22" fontWeight="800" fill="white" textAnchor="middle">{level}</text>

      <text x="100" y="45" fontFamily="system-ui, -apple-system, sans-serif" fontSize="20" fontWeight="700" fill={c.text}>{username}</text>
      <text x="100" y="68" fontFamily="system-ui, -apple-system, sans-serif" fontSize="13" fill={c.muted}>Neuron ML Learning</text>

      <line x1="20" y1="105" x2="400" y2="105" stroke={c.line} strokeWidth="1" strokeDasharray="4,4"/>

      <g transform="translate(25, 125)">
        <rect width="85" height="55" rx="10" fill={c.card}/>
        <rect x="0" y="0" width="85" height="4" rx="2" fill={`url(#card-grad-${theme})`}/>
        <text x="42.5" y="28" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{xp}</text>
        <text x="42.5" y="46" fontFamily="system-ui, -apple-system, sans-serif" fontSize="10" fontWeight="500" fill={c.muted} textAnchor="middle">Total XP</text>
      </g>

      <g transform="translate(120, 125)">
        <rect width="85" height="55" rx="10" fill={c.card}/>
        <rect x="0" y="0" width="85" height="4" rx="2" fill={`url(#card-grad-${theme})`}/>
        <text x="42.5" y="28" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{streak}</text>
        <text x="42.5" y="46" fontFamily="system-ui, -apple-system, sans-serif" fontSize="10" fontWeight="500" fill={c.muted} textAnchor="middle">Day Streak</text>
      </g>

      <g transform="translate(215, 125)">
        <rect width="85" height="55" rx="10" fill={c.card}/>
        <rect x="0" y="0" width="85" height="4" rx="2" fill={`url(#card-grad-${theme})`}/>
        <text x="42.5" y="28" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{lessons}</text>
        <text x="42.5" y="46" fontFamily="system-ui, -apple-system, sans-serif" fontSize="10" fontWeight="500" fill={c.muted} textAnchor="middle">Lessons</text>
      </g>

      <g transform="translate(310, 125)">
        <rect width="85" height="55" rx="10" fill={c.card}/>
        <rect x="0" y="0" width="85" height="4" rx="2" fill={`url(#card-grad-${theme})`}/>
        <text x="42.5" y="28" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{courses}</text>
        <text x="42.5" y="46" fontFamily="system-ui, -apple-system, sans-serif" fontSize="10" fontWeight="500" fill={c.muted} textAnchor="middle">Courses</text>
      </g>
    </svg>
  )
}

export function getcardsvgstring({ username = 'Learner', level = '1', xp = '0', streak = '0', lessons = '0', courses = '0', theme = 'light' }) {
  const colors = {
    light: {
      bg: '#ffffff',
      bgGrad: '#f8fafc',
      card: '#f1f5f9',
      text: '#1e293b',
      muted: '#64748b',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#cbd5e1',
      node: '#94a3b8'
    },
    dark: {
      bg: '#0f172a',
      bgGrad: '#1e293b',
      card: '#334155',
      text: '#f8fafc',
      muted: '#94a3b8',
      accent: '#10b981',
      accentDark: '#059669',
      line: '#475569',
      node: '#64748b'
    }
  }

  const c = colors[theme] || colors.light
  const uid = `card-${theme}-${Date.now()}`

  return `<svg xmlns="http://www.w3.org/2000/svg" width="420" height="200" viewBox="0 0 420 200">
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

    <rect width="420" height="200" rx="16" fill="url(#bg-${uid})" stroke="${c.line}" stroke-width="1"/>

    <g opacity="0.15">
      <line x1="320" y1="30" x2="350" y2="50" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="350" y1="50" x2="380" y2="35" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="350" y1="50" x2="370" y2="75" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="380" y1="35" x2="400" y2="55" stroke="${c.accent}" stroke-width="1.5"/>
      <circle cx="320" cy="30" r="4" fill="${c.accent}"/>
      <circle cx="350" cy="50" r="5" fill="${c.accent}"/>
      <circle cx="380" cy="35" r="4" fill="${c.accent}"/>
      <circle cx="370" cy="75" r="3" fill="${c.accent}"/>
      <circle cx="400" cy="55" r="3" fill="${c.accent}"/>
    </g>

    <g opacity="0.1">
      <line x1="30" y1="160" x2="60" y2="175" stroke="${c.node}" stroke-width="1"/>
      <line x1="60" y1="175" x2="45" y2="190" stroke="${c.node}" stroke-width="1"/>
      <circle cx="30" cy="160" r="3" fill="${c.node}"/>
      <circle cx="60" cy="175" r="4" fill="${c.node}"/>
      <circle cx="45" cy="190" r="3" fill="${c.node}"/>
    </g>

    <circle cx="50" cy="55" r="32" fill="url(#grad-${uid})"/>
    <g transform="translate(50, 55)">
      <circle cx="-8" cy="-8" r="3" fill="white" opacity="0.3"/>
      <circle cx="0" cy="-12" r="3" fill="white" opacity="0.3"/>
      <circle cx="8" cy="-6" r="3" fill="white" opacity="0.3"/>
      <circle cx="-10" cy="4" r="3" fill="white" opacity="0.3"/>
      <circle cx="0" cy="0" r="4" fill="white" opacity="0.5"/>
      <circle cx="10" cy="6" r="3" fill="white" opacity="0.3"/>
      <circle cx="-6" cy="12" r="3" fill="white" opacity="0.3"/>
      <circle cx="6" cy="10" r="3" fill="white" opacity="0.3"/>
      <line x1="-8" y1="-8" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
      <line x1="0" y1="-12" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
      <line x1="8" y1="-6" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
      <line x1="-10" y1="4" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
      <line x1="10" y1="6" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
      <line x1="-6" y1="12" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
      <line x1="6" y1="10" x2="0" y2="0" stroke="white" stroke-width="1" opacity="0.4"/>
    </g>
    <text x="50" y="62" font-family="system-ui, -apple-system, sans-serif" font-size="22" font-weight="800" fill="white" text-anchor="middle">${level}</text>

    <text x="100" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="20" font-weight="700" fill="${c.text}">${username}</text>
    <text x="100" y="68" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="${c.muted}">Neuron ML Learning</text>

    <line x1="20" y1="105" x2="400" y2="105" stroke="${c.line}" stroke-width="1" stroke-dasharray="4,4"/>

    <g transform="translate(25, 125)">
      <rect width="85" height="55" rx="10" fill="${c.card}"/>
      <rect x="0" y="0" width="85" height="4" rx="2" fill="url(#grad-${uid})"/>
      <text x="42.5" y="28" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${xp}</text>
      <text x="42.5" y="46" font-family="system-ui, -apple-system, sans-serif" font-size="10" font-weight="500" fill="${c.muted}" text-anchor="middle">Total XP</text>
    </g>

    <g transform="translate(120, 125)">
      <rect width="85" height="55" rx="10" fill="${c.card}"/>
      <rect x="0" y="0" width="85" height="4" rx="2" fill="url(#grad-${uid})"/>
      <text x="42.5" y="28" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${streak}</text>
      <text x="42.5" y="46" font-family="system-ui, -apple-system, sans-serif" font-size="10" font-weight="500" fill="${c.muted}" text-anchor="middle">Day Streak</text>
    </g>

    <g transform="translate(215, 125)">
      <rect width="85" height="55" rx="10" fill="${c.card}"/>
      <rect x="0" y="0" width="85" height="4" rx="2" fill="url(#grad-${uid})"/>
      <text x="42.5" y="28" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${lessons}</text>
      <text x="42.5" y="46" font-family="system-ui, -apple-system, sans-serif" font-size="10" font-weight="500" fill="${c.muted}" text-anchor="middle">Lessons</text>
    </g>

    <g transform="translate(310, 125)">
      <rect width="85" height="55" rx="10" fill="${c.card}"/>
      <rect x="0" y="0" width="85" height="4" rx="2" fill="url(#grad-${uid})"/>
      <text x="42.5" y="28" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${courses}</text>
      <text x="42.5" y="46" font-family="system-ui, -apple-system, sans-serif" font-size="10" font-weight="500" fill="${c.muted}" text-anchor="middle">Courses</text>
    </g>
  </svg>`
}

export function getcarddataurl(props) {
  const svg = getcardsvgstring(props)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

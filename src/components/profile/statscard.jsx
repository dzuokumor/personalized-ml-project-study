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
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200">
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

      <rect width="400" height="200" rx="12" fill={`url(#card-bg-${theme})`} stroke={c.line} strokeWidth="1"/>

      <g opacity="0.15">
        <line x1="300" y1="30" x2="330" y2="50" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="330" y1="50" x2="360" y2="35" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="330" y1="50" x2="350" y2="75" stroke={c.accent} strokeWidth="1.5"/>
        <line x1="360" y1="35" x2="380" y2="55" stroke={c.accent} strokeWidth="1.5"/>
        <circle cx="300" cy="30" r="4" fill={c.accent}/>
        <circle cx="330" cy="50" r="5" fill={c.accent}/>
        <circle cx="360" cy="35" r="4" fill={c.accent}/>
        <circle cx="350" cy="75" r="3" fill={c.accent}/>
        <circle cx="380" cy="55" r="3" fill={c.accent}/>
      </g>

      <circle cx="50" cy="55" r="30" fill={`url(#card-grad-${theme})`}/>
      <g transform="translate(50, 55)">
        <circle cx="-8" cy="-8" r="4" fill="white" opacity="0.9"/>
        <circle cx="8" cy="-8" r="4" fill="white" opacity="0.9"/>
        <circle cx="0" cy="8" r="4" fill="white" opacity="0.9"/>
        <line x1="-8" y1="-8" x2="8" y2="-8" stroke="white" strokeWidth="1.5" opacity="0.7"/>
        <line x1="-8" y1="-8" x2="0" y2="8" stroke="white" strokeWidth="1.5" opacity="0.7"/>
        <line x1="8" y1="-8" x2="0" y2="8" stroke="white" strokeWidth="1.5" opacity="0.7"/>
      </g>

      <text x="95" y="45" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.text}>{username}</text>
      <text x="95" y="65" fontFamily="system-ui, -apple-system, sans-serif" fontSize="12" fill={c.muted}>Neuron ML Learning</text>
      <text x="95" y="82" fontFamily="system-ui, -apple-system, sans-serif" fontSize="11" fontWeight="600" fill={c.accent}>Level {level}</text>

      <line x1="20" y1="110" x2="380" y2="110" stroke={c.line} strokeWidth="1" strokeDasharray="4 4"/>

      <g transform="translate(20, 125)">
        <rect width="85" height="55" rx="10" fill={c.statBg}/>
        <text x="42.5" y="25" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{xp}</text>
        <text x="42.5" y="42" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="600" fill={c.muted} textAnchor="middle">TOTAL XP</text>
      </g>

      <g transform="translate(115, 125)">
        <rect width="85" height="55" rx="10" fill={c.statBg}/>
        <text x="42.5" y="25" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{streak}</text>
        <text x="42.5" y="42" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="600" fill={c.muted} textAnchor="middle">DAY STREAK</text>
      </g>

      <g transform="translate(210, 125)">
        <rect width="85" height="55" rx="10" fill={c.statBg}/>
        <text x="42.5" y="25" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{lessons}</text>
        <text x="42.5" y="42" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="600" fill={c.muted} textAnchor="middle">LESSONS</text>
      </g>

      <g transform="translate(305, 125)">
        <rect width="75" height="55" rx="10" fill={c.statBg}/>
        <text x="37.5" y="25" fontFamily="system-ui, -apple-system, sans-serif" fontSize="18" fontWeight="700" fill={c.accent} textAnchor="middle">{courses}</text>
        <text x="37.5" y="42" fontFamily="system-ui, -apple-system, sans-serif" fontSize="9" fontWeight="600" fill={c.muted} textAnchor="middle">COURSES</text>
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

  return `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200">
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

    <rect width="400" height="200" rx="12" fill="url(#bg-${uid})" stroke="${c.line}" stroke-width="1"/>

    <g opacity="0.15">
      <line x1="300" y1="30" x2="330" y2="50" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="330" y1="50" x2="360" y2="35" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="330" y1="50" x2="350" y2="75" stroke="${c.accent}" stroke-width="1.5"/>
      <line x1="360" y1="35" x2="380" y2="55" stroke="${c.accent}" stroke-width="1.5"/>
      <circle cx="300" cy="30" r="4" fill="${c.accent}"/>
      <circle cx="330" cy="50" r="5" fill="${c.accent}"/>
      <circle cx="360" cy="35" r="4" fill="${c.accent}"/>
      <circle cx="350" cy="75" r="3" fill="${c.accent}"/>
      <circle cx="380" cy="55" r="3" fill="${c.accent}"/>
    </g>

    <circle cx="50" cy="55" r="30" fill="url(#grad-${uid})"/>
    <g transform="translate(50, 55)">
      <circle cx="-8" cy="-8" r="4" fill="white" opacity="0.9"/>
      <circle cx="8" cy="-8" r="4" fill="white" opacity="0.9"/>
      <circle cx="0" cy="8" r="4" fill="white" opacity="0.9"/>
      <line x1="-8" y1="-8" x2="8" y2="-8" stroke="white" stroke-width="1.5" opacity="0.7"/>
      <line x1="-8" y1="-8" x2="0" y2="8" stroke="white" stroke-width="1.5" opacity="0.7"/>
      <line x1="8" y1="-8" x2="0" y2="8" stroke="white" stroke-width="1.5" opacity="0.7"/>
    </g>

    <text x="95" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.text}">${username}</text>
    <text x="95" y="65" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="${c.muted}">Neuron ML Learning</text>
    <text x="95" y="82" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="600" fill="${c.accent}">Level ${level}</text>

    <line x1="20" y1="110" x2="380" y2="110" stroke="${c.line}" stroke-width="1" stroke-dasharray="4 4"/>

    <g transform="translate(20, 125)">
      <rect width="85" height="55" rx="10" fill="${c.statBg}"/>
      <text x="42.5" y="25" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${xp}</text>
      <text x="42.5" y="42" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="600" fill="${c.muted}" text-anchor="middle">TOTAL XP</text>
    </g>

    <g transform="translate(115, 125)">
      <rect width="85" height="55" rx="10" fill="${c.statBg}"/>
      <text x="42.5" y="25" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${streak}</text>
      <text x="42.5" y="42" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="600" fill="${c.muted}" text-anchor="middle">DAY STREAK</text>
    </g>

    <g transform="translate(210, 125)">
      <rect width="85" height="55" rx="10" fill="${c.statBg}"/>
      <text x="42.5" y="25" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${lessons}</text>
      <text x="42.5" y="42" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="600" fill="${c.muted}" text-anchor="middle">LESSONS</text>
    </g>

    <g transform="translate(305, 125)">
      <rect width="75" height="55" rx="10" fill="${c.statBg}"/>
      <text x="37.5" y="25" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${courses}</text>
      <text x="37.5" y="42" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="600" fill="${c.muted}" text-anchor="middle">COURSES</text>
    </g>
  </svg>`
}

export function getcarddataurl(props) {
  const svg = getcardsvgstring(props)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

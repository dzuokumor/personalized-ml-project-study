export default function statsbadge({ type = 'level', value = '1', theme = 'light' }) {
  const colors = {
    light: { bg: '#ffffff', text: '#1e293b', accent: '#10b981', border: '#e2e8f0' },
    dark: { bg: '#1e293b', text: '#f8fafc', accent: '#10b981', border: '#334155' }
  }

  const c = colors[theme] || colors.light

  const labels = {
    course: 'Course Completed',
    level: 'Level',
    xp: 'XP',
    streak: 'Day Streak'
  }
  const label = labels[type] || type.charAt(0).toUpperCase() + type.slice(1)

  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="200" height="28" viewBox="0 0 200 28">
      <rect width="200" height="28" rx="6" fill={c.bg} stroke={c.border} strokeWidth="1"/>
      <rect x="0" y="0" width="100" height="28" rx="6" fill={c.accent}/>
      <rect x="94" y="0" width="6" height="28" fill={c.accent}/>
      <text x="50" y="18" fontFamily="system-ui, sans-serif" fontSize="11" fontWeight="600" fill="white" textAnchor="middle">{label}</text>
      <text x="150" y="18" fontFamily="system-ui, sans-serif" fontSize="11" fontWeight="600" fill={c.text} textAnchor="middle">{value}</text>
    </svg>
  )
}

export function getbadgesvgstring(type, value, theme = 'light') {
  const colors = {
    light: { bg: '#ffffff', text: '#1e293b', accent: '#10b981', border: '#e2e8f0' },
    dark: { bg: '#1e293b', text: '#f8fafc', accent: '#10b981', border: '#334155' }
  }

  const c = colors[theme] || colors.light

  const labels = {
    course: 'Course Completed',
    level: 'Level',
    xp: 'XP',
    streak: 'Day Streak'
  }
  const label = labels[type] || type.charAt(0).toUpperCase() + type.slice(1)

  return `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="28" viewBox="0 0 200 28">
    <rect width="200" height="28" rx="6" fill="${c.bg}" stroke="${c.border}" stroke-width="1"/>
    <rect x="0" y="0" width="100" height="28" rx="6" fill="${c.accent}"/>
    <rect x="94" y="0" width="6" height="28" fill="${c.accent}"/>
    <text x="50" y="18" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="white" text-anchor="middle">${label}</text>
    <text x="150" y="18" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="${c.text}" text-anchor="middle">${value}</text>
  </svg>`
}

export function getbadgedataurl(type, value, theme = 'light') {
  const svg = getbadgesvgstring(type, value, theme)
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, unquote

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        path_parts = parsed.path.strip('/').split('/')
        username = unquote(path_parts[2]) if len(path_parts) > 2 else 'Learner'

        theme = query.get('theme', ['light'])[0]
        level = query.get('level', ['1'])[0]
        xp = query.get('xp', ['0'])[0]
        streak = query.get('streak', ['0'])[0]
        lessons = query.get('lessons', ['0'])[0]
        courses = query.get('courses', ['0'])[0]

        colors = {
            'light': {'bg': '#ffffff', 'card': '#f8fafc', 'text': '#1e293b', 'muted': '#64748b', 'accent': '#10b981', 'border': '#e2e8f0'},
            'dark': {'bg': '#0f172a', 'card': '#1e293b', 'text': '#f8fafc', 'muted': '#94a3b8', 'accent': '#10b981', 'border': '#334155'}
        }

        c = colors.get(theme, colors['light'])

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="180" viewBox="0 0 400 180">
    <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:{c['accent']};stop-opacity:0.1"/>
            <stop offset="100%" style="stop-color:{c['accent']};stop-opacity:0"/>
        </linearGradient>
    </defs>
    <rect width="400" height="180" rx="12" fill="{c['bg']}" stroke="{c['border']}" stroke-width="1"/>
    <rect width="400" height="180" rx="12" fill="url(#grad)"/>

    <circle cx="50" cy="50" r="28" fill="{c['accent']}"/>
    <text x="50" y="57" font-family="system-ui, sans-serif" font-size="20" font-weight="700" fill="white" text-anchor="middle">{level}</text>

    <text x="90" y="42" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="{c['text']}">{username}</text>
    <text x="90" y="62" font-family="system-ui, sans-serif" font-size="12" fill="{c['muted']}">Neuron ML Learning</text>

    <line x1="20" y1="90" x2="380" y2="90" stroke="{c['border']}" stroke-width="1"/>

    <g transform="translate(30, 110)">
        <rect width="80" height="50" rx="8" fill="{c['card']}"/>
        <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="{c['accent']}" text-anchor="middle">{xp}</text>
        <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="{c['muted']}" text-anchor="middle">Total XP</text>
    </g>

    <g transform="translate(120, 110)">
        <rect width="80" height="50" rx="8" fill="{c['card']}"/>
        <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="{c['accent']}" text-anchor="middle">{streak}</text>
        <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="{c['muted']}" text-anchor="middle">Day Streak</text>
    </g>

    <g transform="translate(210, 110)">
        <rect width="80" height="50" rx="8" fill="{c['card']}"/>
        <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="{c['accent']}" text-anchor="middle">{lessons}</text>
        <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="{c['muted']}" text-anchor="middle">Lessons</text>
    </g>

    <g transform="translate(300, 110)">
        <rect width="80" height="50" rx="8" fill="{c['card']}"/>
        <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="{c['accent']}" text-anchor="middle">{courses}</text>
        <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="{c['muted']}" text-anchor="middle">Courses</text>
    </g>
</svg>'''

        self.send_response(200)
        self.send_header('Content-Type', 'image/svg+xml')
        self.send_header('Cache-Control', 'public, max-age=300')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(svg.encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.end_headers()

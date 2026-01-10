from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, unquote

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        badge_type = query.get('type', ['level'])[0]
        value = unquote(query.get('value', ['1'])[0])
        theme = query.get('theme', ['light'])[0]

        colors = {
            'light': {'bg': '#ffffff', 'text': '#1e293b', 'accent': '#10b981', 'border': '#e2e8f0'},
            'dark': {'bg': '#1e293b', 'text': '#f8fafc', 'accent': '#10b981', 'border': '#334155'}
        }

        c = colors.get(theme, colors['light'])

        labels = {
            'course': 'Course Completed',
            'level': 'Level',
            'xp': 'XP',
            'streak': 'Day Streak'
        }
        label = labels.get(badge_type, badge_type.title())

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="28" viewBox="0 0 200 28">
    <rect width="200" height="28" rx="6" fill="{c['bg']}" stroke="{c['border']}" stroke-width="1"/>
    <rect x="0" y="0" width="100" height="28" rx="6" fill="{c['accent']}"/>
    <rect x="94" y="0" width="6" height="28" fill="{c['accent']}"/>
    <text x="50" y="18" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="white" text-anchor="middle">{label}</text>
    <text x="150" y="18" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="{c['text']}" text-anchor="middle">{value}</text>
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

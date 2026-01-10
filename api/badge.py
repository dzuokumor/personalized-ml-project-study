from flask import Flask, Response, request

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def badge(path):
    badge_type = request.args.get('type', 'level')
    value = request.args.get('value', '1')
    theme = request.args.get('theme', 'light')

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

    return Response(svg, mimetype='image/svg+xml', headers={
        'Cache-Control': 'public, max-age=300',
        'Access-Control-Allow-Origin': '*'
    })

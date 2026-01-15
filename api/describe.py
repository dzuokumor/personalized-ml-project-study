from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            code = data.get('code', '')
            lessontitle = data.get('lessontitle', '')
            coursetitle = data.get('coursetitle', '')

            prompt = f"""Generate a single concise comment (max 10 words) describing what this Python code does.
The code is from a lesson called "{lessontitle}" in the "{coursetitle}" course.

Code:
{code}

Reply with ONLY the comment text, no # symbol, no quotes, just the description.
Example responses:
- Importing libraries for matrix operations
- Training the linear regression model
- Calculating mean squared error loss
- Visualizing the decision boundary"""

            api_key = os.environ.get('OPENROUTER_API_KEY', '')

            request_data = json.dumps({
                "model": "meta-llama/llama-3.2-3b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 30,
                "temperature": 0.2
            }).encode('utf-8')

            req = urllib.request.Request(
                'https://openrouter.ai/api/v1/chat/completions',
                data=request_data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                    'HTTP-Referer': 'https://personalized-ml-project-study.vercel.app',
                    'X-Title': 'ML Study Platform'
                }
            )

            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode('utf-8'))

            description = result['choices'][0]['message']['content'].strip()
            description = description.replace('#', '').strip()
            if description.startswith('"') and description.endswith('"'):
                description = description[1:-1]
            if description.startswith("'") and description.endswith("'"):
                description = description[1:-1]

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'description': description
            }).encode('utf-8'))

        except Exception as e:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'description': 'Code execution from course lesson'
            }).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

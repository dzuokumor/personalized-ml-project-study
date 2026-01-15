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
            messages = data.get('messages', [])
            context = data.get('context', '')

            system_message = """You are an AI tutor helping students learn machine learning.
You are knowledgeable about all aspects of ML from basic linear regression to advanced deep learning.
Explain concepts clearly with examples. Use mathematical notation when helpful but always explain it.
If code is relevant, provide Python examples using common libraries like numpy, scikit-learn, or pytorch.
Be encouraging but accurate. If you don't know something, say so."""

            if context:
                system_message += f"\n\nCurrent lesson context:\n{context}"

            filtered_messages = [
                msg for msg in messages
                if msg.get('content') and msg.get('content').strip()
            ]
            api_messages = [{"role": "system", "content": system_message}] + filtered_messages

            api_key = os.environ.get('OPENROUTER_API_KEY', '')

            request_data = json.dumps({
                "model": "meta-llama/llama-3.2-3b-instruct",
                "messages": api_messages,
                "max_tokens": 1024,
                "temperature": 0.7
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

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))

            assistant_message = result['choices'][0]['message']['content']

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'content': assistant_message
            }).encode('utf-8'))

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': f'API error: {error_body}'
            }).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': str(e)
            }).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

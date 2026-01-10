from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
import base64
import time

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            token = data.get('token')
            reponame = data.get('reponame')
            description = data.get('description', 'ML project from Neuron learning platform')
            code = data.get('code', '')
            readme = data.get('readme', '')
            isprivate = data.get('isprivate', False)

            if not token:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'GitHub token required'}).encode('utf-8'))
                return

            if not reponame:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Repository name required'}).encode('utf-8'))
                return

            repo_data = json.dumps({
                'name': reponame,
                'description': description,
                'private': isprivate,
                'auto_init': True
            }).encode('utf-8')

            req = urllib.request.Request(
                'https://api.github.com/user/repos',
                data=repo_data,
                headers={
                    'Authorization': f'Bearer {token}',
                    'Accept': 'application/vnd.github.v3+json',
                    'Content-Type': 'application/json',
                    'User-Agent': 'Neuron-ML-Platform'
                }
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    repo_result = json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8')
                try:
                    error_json = json.loads(error_body)
                    error_msg = error_json.get('message', 'Failed to create repository')
                except:
                    error_msg = 'Failed to create repository'

                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': error_msg}).encode('utf-8'))
                return

            time.sleep(1)

            full_name = repo_result.get('full_name', '')
            html_url = repo_result.get('html_url', '')

            files_to_add = []
            if readme:
                files_to_add.append({'path': 'README.md', 'content': readme})
            if code:
                files_to_add.append({'path': 'main.py', 'content': code})

            for file_info in files_to_add:
                try:
                    file_data = json.dumps({
                        'message': f'Add {file_info["path"]} from Neuron',
                        'content': base64.b64encode(file_info['content'].encode('utf-8')).decode('utf-8'),
                        'branch': 'main'
                    }).encode('utf-8')

                    file_req = urllib.request.Request(
                        f'https://api.github.com/repos/{full_name}/contents/{file_info["path"]}',
                        data=file_data,
                        method='PUT',
                        headers={
                            'Authorization': f'Bearer {token}',
                            'Accept': 'application/vnd.github.v3+json',
                            'Content-Type': 'application/json',
                            'User-Agent': 'Neuron-ML-Platform'
                        }
                    )
                    urllib.request.urlopen(file_req, timeout=30)
                except:
                    pass

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': True,
                'url': html_url,
                'fullname': full_name
            }).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

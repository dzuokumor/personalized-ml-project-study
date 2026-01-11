from http.server import BaseHTTPRequestHandler
import json
import sys
import io
import traceback
import signal
import ast

class timeoutexception(Exception):
    pass

def timeout_handler(signum, frame):
    raise timeoutexception("Code execution timed out")

allowed_import_names = [
    'numpy', 'np', 'pandas', 'pd',
    'math', 'random', 'statistics',
    'collections', 'itertools', 'functools',
    'json', 're', 'string', 'textwrap',
    'datetime', 'time', 'calendar',
    'decimal', 'fractions',
    'copy', 'pprint',
    'typing', 'dataclasses',
    'enum', 'operator',
    'heapq', 'bisect',
    'csv', 'base64', 'hashlib', 'hmac',
    'html', 'unicodedata',
    'difflib', 'struct', 'codecs',
    'abc', 'contextlib',
    'warnings', 'traceback',
]

forbidden_imports = [
    'os', 'sys', 'subprocess', 'shutil', 'pathlib',
    'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
    'pickle', 'shelve', 'dbm', 'sqlite3',
    'ctypes', 'multiprocessing', 'threading', 'concurrent',
    'importlib', 'runpy', 'zipimport',
    'code', 'codeop', 'pty', 'tty', 'termios',
    'signal', 'mmap', 'fcntl', 'resource',
    'builtins', 'gc', 'inspect', 'dis', 'ast',
]

forbidden_attrs = [
    '__builtins__', '__code__', '__globals__',
    '__subclasses__', '__mro__', '__bases__'
]

def check_code_safety(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module in forbidden_imports:
                    return False, f"Import of '{module}' is not allowed for security reasons"
                if module not in allowed_import_names:
                    return False, f"Import of '{module}' is not allowed. Allowed: numpy, pandas, math, json, datetime, re, collections, and other safe stdlib modules"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split('.')[0]
                if module in forbidden_imports:
                    return False, f"Import from '{module}' is not allowed for security reasons"
                if module not in allowed_import_names:
                    return False, f"Import from '{module}' is not allowed. Allowed: numpy, pandas, math, json, datetime, re, collections, and other safe stdlib modules"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['exec', 'eval', 'compile', 'open', 'input']:
                    return False, f"Function '{node.func.id}' is not allowed for security reasons"

        elif isinstance(node, ast.Attribute):
            if node.attr in forbidden_attrs:
                return False, f"Access to '{node.attr}' is not allowed for security reasons"

    return True, None

def execute_code(code):
    is_safe, error = check_code_safety(code)
    if not is_safe:
        return {'output': '', 'error': error}

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    safe_builtins = {
        'print': print,
        'len': len,
        'range': range,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'abs': abs,
        'max': max,
        'min': min,
        'sum': sum,
        'sorted': sorted,
        'reversed': reversed,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'round': round,
        'pow': pow,
        'isinstance': isinstance,
        'type': type,
        'True': True,
        'False': False,
        'None': None,
    }

    allowed_modules = {}

    stdlib_modules = [
        'math', 'random', 'statistics',
        'collections', 'itertools', 'functools',
        'json', 're', 'string', 'textwrap',
        'datetime', 'time', 'calendar',
        'decimal', 'fractions',
        'copy', 'pprint',
        'typing', 'dataclasses',
        'enum', 'operator',
        'heapq', 'bisect',
        'csv', 'base64', 'hashlib', 'hmac',
        'html', 'unicodedata',
        'difflib', 'struct', 'codecs',
        'abc', 'contextlib',
        'warnings', 'traceback',
    ]

    for mod_name in stdlib_modules:
        try:
            allowed_modules[mod_name] = __import__(mod_name)
        except ImportError:
            pass

    try:
        import numpy as np
        allowed_modules['numpy'] = np
        allowed_modules['np'] = np
    except ImportError:
        pass

    try:
        import pandas as pd
        allowed_modules['pandas'] = pd
        allowed_modules['pd'] = pd
    except ImportError:
        pass

    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in allowed_modules:
            return allowed_modules[name]

        base_module = name.split('.')[0]
        if base_module in ['numpy', 'np', 'pandas', 'pd'] + stdlib_modules:
            return real_import(name, globals, locals, fromlist, level)

        raise ImportError(f"Import of '{name}' is not allowed. Allowed: numpy, pandas, math, random, json, datetime, collections, itertools, functools, and other safe stdlib modules")

    safe_builtins['__import__'] = safe_import

    exec_globals = {'__builtins__': safe_builtins, **allowed_modules}
    exec_locals = {}

    try:
        exec(code, exec_globals, exec_locals)
        output = sys.stdout.getvalue()
        error = sys.stderr.getvalue()
        return {'output': output, 'error': error if error else None}
    except timeoutexception:
        return {'output': '', 'error': 'Code execution timed out (max 5 seconds)'}
    except Exception as e:
        error_output = sys.stderr.getvalue()
        tb = traceback.format_exc()
        return {'output': '', 'error': f"{error_output}\n{tb}" if error_output else tb}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            code = data.get('code', '')

            if len(code) > 10000:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Code too long (max 10000 characters)'
                }).encode('utf-8'))
                return

            result = execute_code(code)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

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

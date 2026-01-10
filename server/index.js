import express from 'express'
import cors from 'cors'
import { spawn } from 'child_process'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { writeFileSync, unlinkSync, existsSync } from 'fs'
import { randomUUID } from 'crypto'
import dotenv from 'dotenv'

dotenv.config({ path: join(dirname(fileURLToPath(import.meta.url)), '..', '.env') })

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const app = express()
const PORT = process.env.PORT || 3001

app.use(cors())
app.use(express.json())

const tempdir = join(__dirname, 'temp')
if (!existsSync(tempdir)) {
  import('fs').then(fs => fs.mkdirSync(tempdir, { recursive: true }))
}

app.post('/api/execute', async (req, res) => {
  const { code } = req.body

  if (!code || typeof code !== 'string') {
    return res.json({ error: 'No code provided' })
  }

  if (code.length > 10000) {
    return res.json({ error: 'Code too long (max 10000 characters)' })
  }

  const forbidden = [
    'import os', 'from os', '__import__', 'eval(', 'exec(',
    'open(', 'subprocess', 'system(', 'popen', 'spawn',
    'import sys', 'from sys', 'globals(', 'locals(',
    'import shutil', 'from shutil', 'import socket', 'from socket',
    'import requests', 'from requests', 'import urllib', 'from urllib'
  ]

  const codelower = code.toLowerCase()
  for (const pattern of forbidden) {
    if (codelower.includes(pattern.toLowerCase())) {
      return res.json({ error: `Security restriction: "${pattern}" is not allowed` })
    }
  }

  const fileid = randomUUID()
  const filepath = join(tempdir, `${fileid}.py`)

  try {
    writeFileSync(filepath, code, 'utf8')

    const result = await new Promise((resolve) => {
      let stdout = ''
      let stderr = ''
      let killed = false

      const python = spawn('python', [filepath], {
        timeout: 30000,
        maxBuffer: 1024 * 1024
      })

      const timeout = setTimeout(() => {
        killed = true
        python.kill('SIGTERM')
      }, 30000)

      python.stdout.on('data', (data) => {
        stdout += data.toString()
        if (stdout.length > 50000) {
          killed = true
          python.kill('SIGTERM')
        }
      })

      python.stderr.on('data', (data) => {
        stderr += data.toString()
      })

      python.on('close', (exitcode) => {
        clearTimeout(timeout)
        if (killed) {
          resolve({ error: 'Execution timed out or output too large' })
        } else if (stderr) {
          const cleanerror = stderr.replace(filepath, 'script.py').replace(/File ".*?"/g, 'File "script.py"')
          resolve({ error: cleanerror })
        } else {
          resolve({ output: stdout })
        }
      })

      python.on('error', (err) => {
        clearTimeout(timeout)
        resolve({ error: `Failed to execute Python: ${err.message}` })
      })
    })

    res.json(result)
  } catch (err) {
    res.json({ error: `Server error: ${err.message}` })
  } finally {
    try {
      if (existsSync(filepath)) {
        unlinkSync(filepath)
      }
    } catch {}
  }
})

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY

app.post('/api/chat', async (req, res) => {
  const { messages, context } = req.body

  if (!OPENROUTER_API_KEY) {
    return res.json({ error: 'OpenRouter API key not configured' })
  }

  try {
    const systemmessage = {
      role: 'system',
      content: `You are an expert ML/AI tutor helping students learn machine learning concepts.
You provide clear, concise explanations with examples when helpful.
You can explain mathematical concepts, code implementations, and theoretical foundations.
Keep responses focused and educational. Use code examples in Python when relevant.
${context ? `\n\nCurrent lesson context: ${context}` : ''}`
    }

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'http://localhost:5173',
        'X-Title': 'ML Learning Platform'
      },
      body: JSON.stringify({
        model: 'meta-llama/llama-3.2-3b-instruct:free',
        messages: [systemmessage, ...messages],
        max_tokens: 1000,
        temperature: 0.7
      })
    })

    const data = await response.json()

    if (data.error) {
      return res.json({ error: data.error.message || 'API error' })
    }

    const content = data.choices?.[0]?.message?.content || 'No response generated'
    res.json({ content })
  } catch (err) {
    res.json({ error: `Failed to connect to AI service: ${err.message}` })
  }
})

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

app.post('/api/github/create-repo', async (req, res) => {
  const { token, reponame, description, code, readme, isprivate } = req.body

  if (!token) {
    return res.json({ error: 'GitHub token required' })
  }

  if (!reponame) {
    return res.json({ error: 'Repository name required' })
  }

  try {
    const createreporesponse = await fetch('https://api.github.com/user/repos', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: reponame,
        description: description || 'ML project from Neuron learning platform',
        private: isprivate || false,
        auto_init: true
      })
    })

    const repodata = await createreporesponse.json()

    if (repodata.errors || repodata.message) {
      return res.json({ error: repodata.message || repodata.errors?.[0]?.message || 'Failed to create repository' })
    }

    await new Promise(r => setTimeout(r, 1000))

    const files = []

    if (readme) {
      files.push({ path: 'README.md', content: readme })
    }

    if (code) {
      files.push({ path: 'main.py', content: code })
    }

    for (const file of files) {
      await fetch(`https://api.github.com/repos/${repodata.full_name}/contents/${file.path}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/vnd.github.v3+json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: `Add ${file.path} from Neuron`,
          content: Buffer.from(file.content).toString('base64'),
          branch: 'main'
        })
      })
    }

    res.json({
      success: true,
      url: repodata.html_url,
      fullname: repodata.full_name
    })
  } catch (err) {
    res.json({ error: `Failed to create repository: ${err.message}` })
  }
})

app.get('/api/badge', (req, res) => {
  const { type = 'level', value = '1', theme } = req.query

  const colors = {
    light: { bg: '#ffffff', text: '#1e293b', accent: '#10b981', border: '#e2e8f0' },
    dark: { bg: '#1e293b', text: '#f8fafc', accent: '#10b981', border: '#334155' }
  }

  const c = colors[theme] || colors.light

  let label = 'Neuron'
  let status = value

  switch (type) {
    case 'course':
      label = 'Course Completed'
      break
    case 'level':
      label = 'Level'
      break
    case 'xp':
      label = 'XP'
      break
    case 'streak':
      label = 'Day Streak'
      break
    default:
      label = type
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="28" viewBox="0 0 200 28">
    <rect width="200" height="28" rx="6" fill="${c.bg}" stroke="${c.border}" stroke-width="1"/>
    <rect x="0" y="0" width="100" height="28" rx="6" fill="${c.accent}"/>
    <rect x="94" y="0" width="6" height="28" fill="${c.accent}"/>
    <text x="50" y="18" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="white" text-anchor="middle">${label}</text>
    <text x="150" y="18" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="${c.text}" text-anchor="middle">${status}</text>
  </svg>`

  res.setHeader('Content-Type', 'image/svg+xml')
  res.setHeader('Cache-Control', 'public, max-age=300')
  res.send(svg)
})

app.get('/api/card', async (req, res) => {
  const { username = 'Learner', theme, level, xp, streak, lessons, courses } = req.query

  const colors = {
    light: { bg: '#ffffff', card: '#f8fafc', text: '#1e293b', muted: '#64748b', accent: '#10b981', border: '#e2e8f0' },
    dark: { bg: '#0f172a', card: '#1e293b', text: '#f8fafc', muted: '#94a3b8', accent: '#10b981', border: '#334155' }
  }

  const c = colors[theme] || colors.light

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="180" viewBox="0 0 400 180">
    <defs>
      <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:${c.accent};stop-opacity:0.1"/>
        <stop offset="100%" style="stop-color:${c.accent};stop-opacity:0"/>
      </linearGradient>
    </defs>
    <rect width="400" height="180" rx="12" fill="${c.bg}" stroke="${c.border}" stroke-width="1"/>
    <rect width="400" height="180" rx="12" fill="url(#grad)"/>

    <circle cx="50" cy="50" r="28" fill="${c.accent}"/>
    <text x="50" y="57" font-family="system-ui, sans-serif" font-size="20" font-weight="700" fill="white" text-anchor="middle">${level || '1'}</text>

    <text x="90" y="42" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="${c.text}">${username || 'Learner'}</text>
    <text x="90" y="62" font-family="system-ui, sans-serif" font-size="12" fill="${c.muted}">Neuron ML Learning</text>

    <line x1="20" y1="90" x2="380" y2="90" stroke="${c.border}" stroke-width="1"/>

    <g transform="translate(30, 110)">
      <rect width="80" height="50" rx="8" fill="${c.card}"/>
      <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${xp || '0'}</text>
      <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="${c.muted}" text-anchor="middle">Total XP</text>
    </g>

    <g transform="translate(120, 110)">
      <rect width="80" height="50" rx="8" fill="${c.card}"/>
      <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${streak || '0'}</text>
      <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="${c.muted}" text-anchor="middle">Day Streak</text>
    </g>

    <g transform="translate(210, 110)">
      <rect width="80" height="50" rx="8" fill="${c.card}"/>
      <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${lessons || '0'}</text>
      <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="${c.muted}" text-anchor="middle">Lessons</text>
    </g>

    <g transform="translate(300, 110)">
      <rect width="80" height="50" rx="8" fill="${c.card}"/>
      <text x="40" y="22" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="${c.accent}" text-anchor="middle">${courses || '0'}</text>
      <text x="40" y="40" font-family="system-ui, sans-serif" font-size="10" fill="${c.muted}" text-anchor="middle">Courses</text>
    </g>
  </svg>`

  res.setHeader('Content-Type', 'image/svg+xml')
  res.setHeader('Cache-Control', 'public, max-age=300')
  res.send(svg)
})

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`)
})

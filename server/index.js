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

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`)
})

const apiurl = import.meta.env.PROD ? '' : 'http://localhost:3001'

export const createrepo = async ({ token, reponame, description, code, readme, isprivate }) => {
  try {
    const response = await fetch(`${apiurl}/api/github/create-repo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        token,
        reponame,
        description,
        code,
        readme,
        isprivate
      })
    })

    return await response.json()
  } catch (err) {
    return { error: err.message }
  }
}

export const generateprojectreadme = ({ coursename, lessonname, description, code }) => {
  return `# ${lessonname}

> From the **${coursename}** course on [Neuron ML Learning Platform](https://neuron-ml.vercel.app)

## Overview

${description || 'A machine learning project completed as part of the Neuron curriculum.'}

## Code

\`\`\`python
${code}
\`\`\`

## About This Project

This project was completed as part of the Neuron ML curriculum, demonstrating practical machine learning skills.

---

*Built with [Neuron](https://neuron-ml.vercel.app) - Learn ML by building real projects*
`
}

export const getbadgeurl = (type, value, theme = 'light') => {
  const baseurl = import.meta.env.PROD
    ? 'https://neuron-ml.vercel.app'
    : 'http://localhost:3001'
  const params = new URLSearchParams({
    type,
    value: String(value),
    theme
  })
  return `${baseurl}/api/badge?${params}`
}

export const getcardurl = ({ username, level, xp, streak, lessons, courses, theme = 'light' }) => {
  const baseurl = import.meta.env.PROD
    ? 'https://neuron-ml.vercel.app'
    : 'http://localhost:3001'
  const params = new URLSearchParams({
    username: username || 'Learner',
    level: level || '1',
    xp: xp || '0',
    streak: streak || '0',
    lessons: lessons || '0',
    courses: courses || '0',
    theme
  })
  return `${baseurl}/api/card?${params}`
}

export const generatebadgemarkdown = (type, value, theme = 'light') => {
  const url = getbadgeurl(type, value, theme)
  return `![${type}: ${value}](${url})`
}

export const generatecardmarkdown = (props) => {
  const url = getcardurl(props)
  return `[![Neuron ML Stats](${url})](https://neuron-ml.vercel.app)`
}

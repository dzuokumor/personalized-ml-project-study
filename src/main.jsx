import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Authprovider } from './contexts/authcontext'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Authprovider>
      <App />
    </Authprovider>
  </StrictMode>
)

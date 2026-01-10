import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Authprovider } from './contexts/authcontext'
import Syncprovider from './components/sync/syncprovider'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Authprovider>
      <Syncprovider>
        <App />
      </Syncprovider>
    </Authprovider>
  </StrictMode>
)

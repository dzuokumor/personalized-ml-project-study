import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

function neuronnode({ position, color, size = 0.08 }) {
  const meshref = useRef()

  useFrame((state) => {
    if (meshref.current) {
      meshref.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.1)
    }
  })

  return (
    <mesh ref={meshref} position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.5}
        metalness={0.3}
        roughness={0.4}
      />
    </mesh>
  )
}

function neuronconnection({ start, end, color }) {
  const points = useMemo(() => {
    return [new THREE.Vector3(...start), new THREE.Vector3(...end)]
  }, [start, end])

  const linegeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    return geometry
  }, [points])

  return (
    <line geometry={linegeometry}>
      <lineBasicMaterial color={color} transparent opacity={0.6} linewidth={1} />
    </line>
  )
}

function rotatingneuroncube() {
  const groupref = useRef()

  const nodes = useMemo(() => [
    [-0.3, -0.3, -0.3],
    [0.3, -0.3, -0.3],
    [0.3, 0.3, -0.3],
    [-0.3, 0.3, -0.3],
    [-0.3, -0.3, 0.3],
    [0.3, -0.3, 0.3],
    [0.3, 0.3, 0.3],
    [-0.3, 0.3, 0.3],
    [0, 0, 0]
  ], [])

  const connections = useMemo(() => [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [0, 8], [1, 8], [2, 8], [3, 8],
    [4, 8], [5, 8], [6, 8], [7, 8]
  ], [])

  useFrame((state) => {
    if (groupref.current) {
      groupref.current.rotation.y = state.clock.elapsedTime * 0.5
      groupref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.3) * 0.2
      groupref.current.position.y = Math.sin(state.clock.elapsedTime * 1.5) * 0.05
    }
  })

  return (
    <group ref={groupref}>
      {nodes.map((pos, i) => (
        <neuronnode
          key={i}
          position={pos}
          color={i === 8 ? '#fbbf24' : i % 2 === 0 ? '#10b981' : '#06b6d4'}
          size={i === 8 ? 0.12 : 0.07}
        />
      ))}
      {connections.map(([startidx, endidx], i) => (
        <neuronconnection
          key={i}
          start={nodes[startidx]}
          end={nodes[endidx]}
          color={startidx === 8 || endidx === 8 ? '#fbbf24' : '#10b981'}
        />
      ))}
    </group>
  )
}

export default function neuroncrown() {
  return (
    <div className="w-20 h-20">
      <Canvas camera={{ position: [0, 0, 2], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#10b981" />
        <rotatingneuroncube />
      </Canvas>
    </div>
  )
}

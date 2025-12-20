export default function ProgressBar({ value, size = 'md' }) {
  const heights = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3'
  }

  return (
    <div className={`w-full bg-slate-200 rounded-full overflow-hidden ${heights[size]}`}>
      <div
        className="h-full bg-emerald-600 rounded-full transition-all duration-300"
        style={{ width: `${value}%` }}
      />
    </div>
  )
}

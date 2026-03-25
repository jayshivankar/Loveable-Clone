export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-[var(--background)] text-[var(--foreground)] overflow-hidden">
      {/* Sidebar Placeholder */}
      <aside className="w-64 border-r border-slate-800 bg-slate-900/50 flex flex-col shrink-0">
        <div className="p-4 border-b border-slate-800 font-bold tracking-tight">CodeForge Sessions</div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <div className="p-2 rounded bg-blue-600/20 text-blue-400 text-sm cursor-pointer border border-blue-500/30">Current Project</div>
          <div className="p-2 rounded hover:bg-slate-800 text-slate-400 text-sm cursor-pointer transition-colors">Previous Code Gen</div>
        </div>
        <div className="p-4 border-t border-slate-800">
          <button className="w-full py-2 bg-slate-800 hover:bg-slate-700 rounded text-sm transition-colors">+ New Session</button>
        </div>
      </aside>
      
      {/* Main Content */}
      <main className="flex-1 flex flex-col relative w-full h-full">
        {children}
      </main>
    </div>
  );
}

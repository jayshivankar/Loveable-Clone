"use client";

import { useState, useRef, useEffect } from "react";
import { TaskPlanReview, TaskPlan } from "../../components/TaskPlanReview";

export default function WorkspacePage() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<{role: 'user' | 'assistant', content: string}[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const [approvalNeeded, setApprovalNeeded] = useState(false);
  const [taskPlan, setTaskPlan] = useState<TaskPlan | null>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isGenerating, approvalNeeded]);

  const handleSSEEvent = (data: any, resolveCompletion: () => void) => {
    setMessages(prev => {
      const newMessages = [...prev];
      const lastMsg = newMessages[newMessages.length - 1];
      if (data.event === "node.started") {
         lastMsg.content += `\n> 🔄 Started: ${data.node}`;
      } else if (data.event === "run.completed") {
         lastMsg.content += `\n\n✅ Generation Complete.`;
      } else if (data.event === "hitl.required") {
         lastMsg.content += `\n\n⏸️ Task Plan generated. Awaiting approval...`;
      } else if (data.event === "hitl.resumed") {
         lastMsg.content += `\n\n▶️ Resumed execution (${data.action})`;
      }
      return newMessages;
    });

    if (data.event === "hitl.required") {
       setTaskPlan(data.task_plan);
       setApprovalNeeded(true);
       setIsGenerating(false);
       resolveCompletion();
    } else if (data.event === "run.completed") {
       setIsGenerating(false);
       resolveCompletion();
    }
  };

  const readFetchStream = async (response: Response) => {
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = "";

    return new Promise<void>(async (resolve) => {
        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (!dataStr) continue;
                        try {
                            const data = JSON.parse(dataStr);
                            handleSSEEvent(data, resolve);
                        } catch (e) {
                            console.error('SSE parse error', e);
                        }
                    }
                }
            }
        } finally {
            resolve();
        }
    });
  };

  const handleResume = async (action: 'approve' | 'edit', editedPlan?: TaskPlan) => {
    if (!currentRunId) return;
    setIsSubmitting(true);
    
    try {
        const payload: any = { action };
        if (editedPlan) {
            payload.edited_plan = editedPlan;
        }
        
        const response = await fetch(`http://localhost:8000/api/v1/runs/${currentRunId}/resume`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        setIsSubmitting(false);
        setApprovalNeeded(false);
        setTaskPlan(null);
        setIsGenerating(true);
        
        await readFetchStream(response);
    } catch (err) {
        console.error("Resume error", err);
        setIsSubmitting(false);
        setIsGenerating(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isGenerating) return;

    setMessages(prev => [...prev, { role: 'user', content: prompt }]);
    setPrompt("");
    setIsGenerating(true);
    setApprovalNeeded(false);
    setTaskPlan(null);

    try {
      // 1. Create Run
      const createRes = await fetch("http://localhost:8000/api/v1/runs/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: "test", prompt: prompt })
      });
      const runData = await createRes.json();
      setCurrentRunId(runData.id);

      // 2. Stream
      const eventSource = new EventSource(`http://localhost:8000/api/v1/runs/${runData.id}/stream?prompt=${encodeURIComponent(prompt)}`);
      
      setMessages(prev => [...prev, { role: 'assistant', content: "" }]);

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleSSEEvent(data, () => eventSource.close());
      };

      eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        eventSource.close();
        setIsGenerating(false);
      };

    } catch (err) {
      console.error(err);
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      <header className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10 flex justify-between items-center">
        <h2 className="text-lg font-semibold">Workspace / <span className="text-slate-500 font-normal">Antigravity Coding</span></h2>
        <div className="px-3 py-1 bg-green-500/20 text-green-400 text-xs rounded-full border border-green-500/30">Backend Connected</div>
      </header>
      
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-12 h-12 opacity-50">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.59 14.37a6 6 0 01-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 006.16-12.12A14.98 14.98 0 009.631 8.41m5.96 5.96a14.926 14.926 0 01-5.841 2.58m-.119-8.54a6 6 0 00-7.381 5.84h4.8m2.581-5.84a14.927 14.927 0 00-2.58 5.84m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 01-2.448-2.448 14.9 14.9 0 01.06-.312m-2.24 2.39a4.493 4.493 0 00-1.757 4.306 4.433 4.433 0 002.906 2.907 4.493 4.493 0 004.306-1.758M16.5 9a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" />
            </svg>
            <p>What are we building today?</p>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`p-4 rounded-2xl max-w-3xl whitespace-pre-wrap font-mono text-sm shadow-sm ${
                msg.role === 'user' ? 'bg-blue-600/90 text-white !rounded-br-sm' : 'glass-panel !rounded-bl-sm border-slate-700/50'
              }`}>
                {msg.content}
              </div>
            </div>
          ))
        )}
        
        {approvalNeeded && taskPlan && (
          <div className="flex justify-start animate-fade-in-up">
            <TaskPlanReview 
                taskPlan={taskPlan} 
                onApprove={() => handleResume('approve')} 
                onEdit={(plan) => handleResume('edit', plan)} 
                isSubmitting={isSubmitting} 
            />
          </div>
        )}
      </div>

      <div className="p-4 bg-[var(--background)] border-t border-slate-800 z-10 w-full">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative flex items-center">
            <input
              type="text"
              className="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-6 py-4 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder-slate-500 pr-32 shadow-inner"
              placeholder="Describe your Next.js frontend requirements..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isGenerating || approvalNeeded}
            />
            <button 
              disabled={!prompt.trim() || isGenerating || approvalNeeded}
              className="absolute right-2 px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 font-semibold rounded-lg transition-colors flex items-center gap-2 h-10"
            >
              {isGenerating ? "Working..." : "Send"}
            </button>
          </form>
          <p className="text-center text-xs text-slate-600 mt-3 flex justify-center gap-6">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500"></span> Backend OK</span>
            <span>{isGenerating ? "Cost Tracker: $0.04" : "Cost Tracker: $0.00"}</span>
            <span>Tokens: {isGenerating ? "14.2k" : "0"} / 200k limit</span>
          </p>
        </div>
      </div>
    </div>
  );
}

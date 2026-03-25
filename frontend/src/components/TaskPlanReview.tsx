import React, { useState } from 'react';

export interface TaskStep {
  filepath: string;
  task_description: string;
  depends_on: string[];
}

export interface TaskPlan {
  implementation_steps: TaskStep[];
}

interface TaskPlanReviewProps {
  taskPlan: TaskPlan;
  onApprove: () => void;
  onEdit: (editedPlan: TaskPlan) => void;
  isSubmitting: boolean;
}

export function TaskPlanReview({ taskPlan, onApprove, onEdit, isSubmitting }: TaskPlanReviewProps) {
  const [isEditMode, setIsEditMode] = useState(false);
  const [steps, setSteps] = useState<TaskStep[]>(taskPlan.implementation_steps || []);
  const [error, setError] = useState<string | null>(null);

  const handleStepChange = (index: number, field: keyof TaskStep, value: string | string[]) => {
    const newSteps = [...steps];
    newSteps[index] = { ...newSteps[index], [field]: value };
    setSteps(newSteps);
  };

  const handleDelete = (index: number) => {
    const newSteps = [...steps];
    newSteps.splice(index, 1);
    setSteps(newSteps);
  };

  const handleAdd = () => {
    setSteps([...steps, { filepath: '', task_description: '', depends_on: [] }]);
  };

  const handleMoveUp = (index: number) => {
    if (index === 0) return;
    const newSteps = [...steps];
    const temp = newSteps[index];
    newSteps[index] = newSteps[index - 1];
    newSteps[index - 1] = temp;
    setSteps(newSteps);
  };

  const handleMoveDown = (index: number) => {
    if (index === steps.length - 1) return;
    const newSteps = [...steps];
    const temp = newSteps[index];
    newSteps[index] = newSteps[index + 1];
    newSteps[index + 1] = temp;
    setSteps(newSteps);
  };

  const validateAndSubmit = () => {
    setError(null);
    for (let i = 0; i < steps.length; i++) {
      if (!steps[i].filepath.trim() || !steps[i].task_description.trim()) {
        setError(`Step ${i + 1} has empty filepath or description.`);
        return;
      }
    }
    onEdit({ implementation_steps: steps });
  };

  return (
    <div className="p-6 rounded-2xl max-w-4xl glass-panel !rounded-bl-sm border-blue-500/50 bg-blue-500/10 w-full relative overflow-hidden text-sm">
      <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-blue-400 font-bold flex items-center gap-2">
           <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
           Architect Implementation Plan
        </h3>
        <button 
          onClick={() => setIsEditMode(!isEditMode)} 
          className="text-xs px-3 py-1 bg-slate-800 text-white rounded hover:bg-slate-700 transition"
          disabled={isSubmitting}
        >
          {isEditMode ? "Cancel Edit" : "Edit Plan"}
        </button>
      </div>

      <div className="space-y-4 mb-6">
        {steps.map((step, idx) => (
          <div key={idx} className="p-3 bg-slate-900/50 rounded border border-slate-700/50">
            {isEditMode ? (
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  <span className="text-slate-500 font-mono mt-2">{idx + 1}.</span>
                  <div className="flex-1 space-y-2">
                    <input 
                      value={step.filepath}
                      onChange={(e) => handleStepChange(idx, 'filepath', e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 outline-none focus:border-blue-500 font-mono text-xs"
                      placeholder="filepath (e.g. src/App.tsx)"
                    />
                    <textarea 
                      value={step.task_description}
                      onChange={(e) => handleStepChange(idx, 'task_description', e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 outline-none focus:border-blue-500 min-h-[60px]"
                      placeholder="Task description..."
                    />
                    <input 
                      value={(step.depends_on || []).join(', ')}
                      onChange={(e) => handleStepChange(idx, 'depends_on', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                      className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 outline-none focus:border-blue-500 font-mono text-xs text-slate-400"
                      placeholder="dependencies (comma separated)"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <button onClick={() => handleMoveUp(idx)} className="p-1 hover:bg-slate-700 rounded" title="Move Up">↑</button>
                    <button onClick={() => handleMoveDown(idx)} className="p-1 hover:bg-slate-700 rounded" title="Move Down">↓</button>
                    <button onClick={() => handleDelete(idx)} className="p-1 hover:bg-red-900/50 text-red-400 rounded mt-auto" title="Delete">×</button>
                  </div>
                </div>
              </div>
            ) : (
               <div>
                  <div className="font-mono text-blue-300 font-semibold mb-1 flex items-start gap-2">
                    <span className="text-slate-500">{idx + 1}.</span> {step.filepath}
                  </div>
                  <div className="pl-6 text-slate-300 whitespace-pre-wrap">{step.task_description}</div>
                  {step.depends_on && step.depends_on.length > 0 && (
                    <div className="pl-6 mt-2 text-xs text-slate-500 font-mono">
                      Depends on: {step.depends_on.join(', ')}
                    </div>
                  )}
               </div>
            )}
          </div>
        ))}
      </div>

      {isEditMode && (
        <div className="mb-6">
          <button onClick={handleAdd} className="w-full py-2 border border-dashed border-slate-600 text-slate-400 rounded hover:bg-slate-800 transition">
            + Add Implementation Step
          </button>
        </div>
      )}

      {error && <div className="text-red-400 mb-4 px-3 py-2 bg-red-900/20 rounded border border-red-900/50">{error}</div>}

      <div className="flex gap-3">
        {isEditMode ? (
          <button 
            disabled={isSubmitting}
            onClick={validateAndSubmit} 
            className="px-4 py-2 bg-blue-600 text-white font-bold rounded shadow hover:bg-blue-500 transition-colors disabled:opacity-50"
          >
            {isSubmitting ? "Submitting..." : "Approve Changes"}
          </button>
        ) : (
          <button 
            disabled={isSubmitting}
            onClick={onApprove} 
            className="px-4 py-2 bg-blue-600 text-white font-bold rounded shadow hover:bg-blue-500 transition-colors disabled:opacity-50"
          >
            {isSubmitting ? "Approving..." : "Approve & Continue"}
          </button>
        )}
      </div>
    </div>
  );
}

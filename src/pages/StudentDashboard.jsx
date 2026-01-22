import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, ChevronDown, ChevronRight, CheckCircle2, Circle, BookOpen, 
  Code, Link, Trophy, Clock, Target, Sparkles, RotateCcw, LogOut,
  User, TrendingUp, Calendar, Award
} from 'lucide-react';
import { useAuth } from '../lib/AuthContext';
import { 
  signOut, getProgress, updateProgress, getTotalTimeSpent,
  saveExerciseAttempt, getExerciseAttempts, logTimeSpent 
} from '../lib/supabase';
import { curriculum, getAllWeeks, getTotalWeeks } from '../lib/curriculum';

// Progress Ring Component
const ProgressRing = ({ progress, size = 60, strokeWidth = 6 }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;
  
  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#e5e7eb" strokeWidth={strokeWidth} />
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#10b981" strokeWidth={strokeWidth}
        strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" className="transition-all duration-500" />
    </svg>
  );
};

// Badge Component
const Badge = ({ children, variant = 'default' }) => {
  const variants = {
    default: 'bg-gray-100 text-gray-700',
    new: 'bg-emerald-100 text-emerald-700',
    revised: 'bg-blue-100 text-blue-700',
    capstone: 'bg-purple-100 text-purple-700'
  };
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>{children}</span>;
};

// Resource Link Component
const ResourceLink = ({ resource }) => {
  const icons = { course: '📚', video: '🎬', docs: '📖', tutorial: '💻', article: '📝', tool: '🔧', paper: '📄' };
  return (
    <a href={resource.url} target="_blank" rel="noopener noreferrer"
      className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-50 transition-colors group">
      <span>{icons[resource.type] || '🔗'}</span>
      <span className="text-sm text-gray-700 group-hover:text-emerald-600">{resource.title}</span>
      <Link className="w-3 h-3 text-gray-400 ml-auto" />
    </a>
  );
};

// Code Exercise Component
const CodeExercise = ({ exercise, userId, onAttempt }) => {
  const [code, setCode] = useState(exercise.starterCode);
  const [showSolution, setShowSolution] = useState(false);
  
  const handleShowSolution = async () => {
    if (!showSolution && userId) {
      await saveExerciseAttempt(userId, exercise.id, code, true);
      onAttempt?.();
    }
    setShowSolution(!showSolution);
  };
  
  return (
    <div className="mt-4 border rounded-lg overflow-hidden">
      <div className="bg-gray-800 text-white px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Code className="w-4 h-4" />
          <span className="font-medium">{exercise.title}</span>
        </div>
        <div className="flex gap-2">
          <button onClick={() => setCode(exercise.starterCode)}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-700 rounded hover:bg-gray-600">
            <RotateCcw className="w-3 h-3" /> Reset
          </button>
          <button onClick={handleShowSolution}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-emerald-600 rounded hover:bg-emerald-500">
            {showSolution ? 'Hide Solution' : 'Show Solution'}
          </button>
        </div>
      </div>
      <p className="px-4 py-2 text-sm text-gray-600 bg-gray-50 border-b">{exercise.description}</p>
      <textarea value={showSolution ? exercise.solution : code}
        onChange={(e) => !showSolution && setCode(e.target.value)}
        className="w-full h-80 p-4 font-mono text-sm bg-gray-900 text-green-400 resize-none focus:outline-none"
        spellCheck={false} readOnly={showSolution} />
    </div>
  );
};

// Week Card Component
const WeekCard = ({ week, completed, onToggle, expanded, onExpand, userId, onExerciseAttempt }) => {
  return (
    <div className="border rounded-lg overflow-hidden bg-white">
      <button onClick={onExpand} className="w-full px-4 py-3 flex items-center gap-3 hover:bg-gray-50 transition-colors">
        <button onClick={(e) => { e.stopPropagation(); onToggle(); }} className="flex-shrink-0">
          {completed ? <CheckCircle2 className="w-6 h-6 text-emerald-500" /> : <Circle className="w-6 h-6 text-gray-300 hover:text-emerald-400" />}
        </button>
        <div className="flex-1 text-left">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`font-medium ${completed ? 'text-gray-500 line-through' : 'text-gray-800'}`}>{week.title}</span>
            {week.isNew && <Badge variant="new">NEW</Badge>}
            {week.isRevised && <Badge variant="revised">REVISED</Badge>}
            {week.hasCapstone && <Badge variant="capstone">★ Capstone Start</Badge>}
          </div>
        </div>
        {expanded ? <ChevronDown className="w-5 h-5 text-gray-400" /> : <ChevronRight className="w-5 h-5 text-gray-400" />}
      </button>
      
      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
              <Target className="w-4 h-4" /> Live Sessions
            </h4>
            <ul className="space-y-1">
              {week.sessions.map((session, i) => (
                <li key={i} className="text-sm text-gray-600 pl-4 border-l-2 border-emerald-200">Session {i + 1}: {session}</li>
              ))}
            </ul>
          </div>
          
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
              <Clock className="w-4 h-4" /> Async Work
            </h4>
            <ul className="space-y-1">
              {week.asyncWork.map((task, i) => (
                <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                  <span className="text-gray-400">•</span>
                  <span className={task.includes('★') ? 'text-purple-600 font-medium' : ''}>{task}</span>
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
              <BookOpen className="w-4 h-4" /> Resources
            </h4>
            <div className="grid gap-1">
              {week.resources.map((resource, i) => <ResourceLink key={i} resource={resource} />)}
            </div>
          </div>
          
          {week.exercise && <CodeExercise exercise={week.exercise} userId={userId} onAttempt={onExerciseAttempt} />}
        </div>
      )}
    </div>
  );
};

// Format time helper
const formatTime = (seconds) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
};

export default function StudentDashboard() {
  const { user, profile, isAdmin } = useAuth();
  const navigate = useNavigate();
  const [completedItems, setCompletedItems] = useState({});
  const [expandedItems, setExpandedItems] = useState({});
  const [activeTab, setActiveTab] = useState('curriculum');
  const [totalTime, setTotalTime] = useState(0);
  const [exerciseCount, setExerciseCount] = useState(0);
  const sessionStartRef = useRef(Date.now());
  const lastLogRef = useRef(Date.now());

  // Load progress from database
  useEffect(() => {
    if (user) {
      loadProgress();
      loadStats();
    }
  }, [user]);

  // Time tracking - log every 5 minutes
  useEffect(() => {
    const interval = setInterval(async () => {
      if (user) {
        const now = Date.now();
        const secondsSpent = Math.floor((now - lastLogRef.current) / 1000);
        if (secondsSpent >= 60) {
          await logTimeSpent(user.id, 'session', secondsSpent);
          lastLogRef.current = now;
          setTotalTime(prev => prev + secondsSpent);
        }
      }
    }, 60000); // Check every minute

    return () => clearInterval(interval);
  }, [user]);

  // Log time on unmount
  useEffect(() => {
    return () => {
      if (user) {
        const secondsSpent = Math.floor((Date.now() - lastLogRef.current) / 1000);
        if (secondsSpent > 10) {
          logTimeSpent(user.id, 'session', secondsSpent);
        }
      }
    };
  }, [user]);

  const loadProgress = async () => {
    const { data } = await getProgress(user.id);
    if (data) {
      const completed = {};
      data.forEach(p => { if (p.completed) completed[p.week_id] = true; });
      setCompletedItems(completed);
    }
  };

  const loadStats = async () => {
    const { data: time } = await getTotalTimeSpent(user.id);
    setTotalTime(time || 0);
    
    const { data: attempts } = await getExerciseAttempts(user.id);
    if (attempts) {
      const uniqueExercises = new Set(attempts.map(a => a.exercise_id));
      setExerciseCount(uniqueExercises.size);
    }
  };

  const toggleComplete = async (id) => {
    const newCompleted = { ...completedItems, [id]: !completedItems[id] };
    setCompletedItems(newCompleted);
    
    await updateProgress(user.id, id, {
      completed: newCompleted[id],
      completed_at: newCompleted[id] ? new Date().toISOString() : null
    });
  };

  const toggleExpand = (id) => {
    setExpandedItems(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const handleSignOut = async () => {
    // Log final time before signing out
    const secondsSpent = Math.floor((Date.now() - lastLogRef.current) / 1000);
    if (secondsSpent > 10) {
      await logTimeSpent(user.id, 'session', secondsSpent);
    }
    await signOut();
    navigate('/');
  };

  // Calculate progress
  const totalItems = getTotalWeeks();
  const completedCount = Object.values(completedItems).filter(Boolean).length;
  const progressPercent = Math.round((completedCount / totalItems) * 100);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-emerald-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">AI Fundamentals Bootcamp</h1>
                <p className="text-sm text-gray-500">Welcome, {profile?.full_name || user?.email}</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {isAdmin && (
                <button onClick={() => navigate('/admin')}
                  className="px-3 py-1.5 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200">
                  Admin Dashboard
                </button>
              )}
              <div className="text-right mr-2">
                <p className="text-sm font-medium text-gray-700">{completedCount}/{totalItems}</p>
                <p className="text-xs text-gray-500">Completed</p>
              </div>
              <div className="relative">
                <ProgressRing progress={progressPercent} />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-sm font-bold text-emerald-600">{progressPercent}%</span>
                </div>
              </div>
              <button onClick={handleSignOut} className="p-2 text-gray-500 hover:text-gray-700">
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Cards */}
      <div className="max-w-5xl mx-auto px-4 py-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-100 rounded-lg">
                <TrendingUp className="w-5 h-5 text-emerald-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{progressPercent}%</p>
                <p className="text-xs text-gray-500">Progress</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Clock className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{formatTime(totalTime)}</p>
                <p className="text-xs text-gray-500">Time Spent</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Code className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{exerciseCount}</p>
                <p className="text-xs text-gray-500">Exercises Done</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-100 rounded-lg">
                <Award className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{completedCount}</p>
                <p className="text-xs text-gray-500">Weeks Complete</p>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b mb-6">
          {[
            { id: 'curriculum', label: 'Curriculum', icon: BookOpen },
            { id: 'exercises', label: 'Exercises', icon: Code },
            { id: 'resources', label: 'All Resources', icon: Link }
          ].map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id ? 'border-emerald-500 text-emerald-600' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}>
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        {activeTab === 'curriculum' && (
          <div className="space-y-6">
            {/* Target Roles Banner */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-4 text-white">
              <div className="flex items-start gap-3">
                <Trophy className="w-6 h-6 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold">Target Roles After Completion</h3>
                  <p className="text-sm opacity-90 mt-1">
                    Junior Data Scientist ($88-110K) • Data Analyst ($57-75K) • Junior ML Analyst ($90-103K)
                  </p>
                </div>
              </div>
            </div>

            {/* Pre-Work */}
            <div>
              <h2 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-emerald-500" />
                Pre-Work (Complete Before Week 1)
              </h2>
              <WeekCard
                week={curriculum.preWork}
                completed={completedItems[curriculum.preWork.id]}
                onToggle={() => toggleComplete(curriculum.preWork.id)}
                expanded={expandedItems[curriculum.preWork.id]}
                onExpand={() => toggleExpand(curriculum.preWork.id)}
                userId={user?.id}
                onExerciseAttempt={loadStats}
              />
            </div>

            {/* Modules */}
            {curriculum.modules.map(module => (
              <div key={module.id}>
                <h2 className="text-lg font-semibold text-gray-800 mb-3">{module.title}</h2>
                <div className="space-y-2">
                  {module.weeks.map(week => (
                    <WeekCard
                      key={week.id}
                      week={week}
                      completed={completedItems[week.id]}
                      onToggle={() => toggleComplete(week.id)}
                      expanded={expandedItems[week.id]}
                      onExpand={() => toggleExpand(week.id)}
                      userId={user?.id}
                      onExerciseAttempt={loadStats}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'exercises' && (
          <div className="space-y-6">
            <p className="text-gray-600">All hands-on coding exercises:</p>
            {curriculum.preWork.exercise && (
              <div>
                <h3 className="font-semibold text-gray-800 mb-2">Pre-Work</h3>
                <CodeExercise exercise={curriculum.preWork.exercise} userId={user?.id} onAttempt={loadStats} />
              </div>
            )}
            {curriculum.modules.map(module => (
              <div key={module.id}>
                <h3 className="font-semibold text-gray-800 mb-2">{module.title}</h3>
                {module.weeks.filter(w => w.exercise).map(week => (
                  <div key={week.id} className="mb-4">
                    <p className="text-sm text-gray-500 mb-1">{week.title}</p>
                    <CodeExercise exercise={week.exercise} userId={user?.id} onAttempt={loadStats} />
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'resources' && (
          <div className="space-y-6">
            <p className="text-gray-600">Complete resource library:</p>
            <div className="bg-white rounded-lg border p-4">
              <h3 className="font-semibold text-gray-800 mb-3">Pre-Work Resources</h3>
              <div className="grid gap-1">
                {curriculum.preWork.resources.map((r, i) => <ResourceLink key={i} resource={r} />)}
              </div>
            </div>
            {curriculum.modules.map(module => (
              <div key={module.id} className="bg-white rounded-lg border p-4">
                <h3 className="font-semibold text-gray-800 mb-3">{module.title}</h3>
                {module.weeks.map(week => (
                  <div key={week.id} className="mb-4 last:mb-0">
                    <p className="text-sm font-medium text-gray-600 mb-2">{week.title}</p>
                    <div className="grid gap-1 pl-4 border-l-2 border-gray-100">
                      {week.resources.map((r, i) => <ResourceLink key={i} resource={r} />)}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>

      <footer className="max-w-5xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
        AI Fundamentals Bootcamp v2.0 • Progress saves automatically
      </footer>
    </div>
  );
}

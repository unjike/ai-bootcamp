import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, ChevronDown, ChevronRight, CheckCircle2, Circle, BookOpen, 
  Link, Clock, Target, Sparkles, RotateCcw, LogOut,
  Lock, Unlock, HelpCircle, Award, TrendingUp, X
} from 'lucide-react';
import { useAuth } from '../lib/AuthContext';
import { 
  signOut, getProgress, updateProgress, getTotalTimeSpent,
  logTimeSpent, saveQuizResult, getQuizResults
} from '../lib/supabase';
import { curriculum, getAllWeeks, getTotalWeeks, quizzes, isWeekUnlocked } from '../lib/curriculum';

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
    capstone: 'bg-purple-100 text-purple-700',
    locked: 'bg-gray-200 text-gray-500',
    passed: 'bg-green-100 text-green-700'
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

// Quiz Component
const QuizModal = ({ quiz, weekId, userId, onComplete, onClose, previousResult }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [score, setScore] = useState(0);

  const handleAnswer = (questionId, answerIndex) => {
    setAnswers({ ...answers, [questionId]: answerIndex });
  };

  const handleSubmit = async () => {
    let correct = 0;
    quiz.questions.forEach(q => {
      if (answers[q.id] === q.correctAnswer) correct++;
    });
    
    const scorePercent = Math.round((correct / quiz.questions.length) * 100);
    const passed = scorePercent >= quiz.passingScore;
    
    setScore(scorePercent);
    setShowResults(true);
    
    // Save to database
    await saveQuizResult(userId, quiz.id, weekId, scorePercent, passed, answers);
    
    if (passed) {
      onComplete(quiz.id, weekId);
    }
  };

  const question = quiz.questions[currentQuestion];
  const allAnswered = quiz.questions.every(q => answers[q.id] !== undefined);

  if (showResults) {
    const passed = score >= quiz.passingScore;
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-2xl max-w-md w-full p-6 text-center">
          <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 ${passed ? 'bg-green-100' : 'bg-red-100'}`}>
            {passed ? (
              <CheckCircle2 className="w-10 h-10 text-green-600" />
            ) : (
              <X className="w-10 h-10 text-red-600" />
            )}
          </div>
          <h3 className="text-2xl font-bold mb-2">{passed ? 'Congratulations!' : 'Keep Learning!'}</h3>
          <p className="text-4xl font-bold mb-2" style={{ color: passed ? '#10b981' : '#ef4444' }}>{score}%</p>
          <p className="text-gray-600 mb-4">
            {passed 
              ? 'You passed the quiz! The next week is now unlocked.' 
              : `You need ${quiz.passingScore}% to pass. Review the material and try again.`}
          </p>
          <button 
            onClick={onClose}
            className="px-6 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
          >
            {passed ? 'Continue' : 'Try Again Later'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between bg-emerald-50">
          <div>
            <h3 className="font-bold text-lg">{quiz.title}</h3>
            <p className="text-sm text-gray-600">Question {currentQuestion + 1} of {quiz.questions.length}</p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-gray-200">
          <div 
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${((currentQuestion + 1) / quiz.questions.length) * 100}%` }}
          />
        </div>

        {/* Question */}
        <div className="p-6 flex-1 overflow-auto">
          <h4 className="text-lg font-medium mb-4">{question.question}</h4>
          <div className="space-y-3">
            {question.options.map((option, index) => (
              <button
                key={index}
                onClick={() => handleAnswer(question.id, index)}
                className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                  answers[question.id] === index 
                    ? 'border-emerald-500 bg-emerald-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <span className="font-medium mr-2">{String.fromCharCode(65 + index)}.</span>
                {option}
              </button>
            ))}
          </div>
        </div>

        {/* Navigation */}
        <div className="p-4 border-t flex justify-between">
          <button
            onClick={() => setCurrentQuestion(Math.max(0, currentQuestion - 1))}
            disabled={currentQuestion === 0}
            className="px-4 py-2 text-gray-600 disabled:opacity-50"
          >
            Previous
          </button>
          
          {currentQuestion < quiz.questions.length - 1 ? (
            <button
              onClick={() => setCurrentQuestion(currentQuestion + 1)}
              disabled={answers[question.id] === undefined}
              className="px-6 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50"
            >
              Next
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!allAnswered}
              className="px-6 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50"
            >
              Submit Quiz
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// Week Card Component
const WeekCard = ({ 
  week, 
  completed, 
  onToggle, 
  expanded, 
  onExpand, 
  userId, 
  onQuizComplete,
  isLocked,
  quizPassed,
  quizResult
}) => {
  const [showQuiz, setShowQuiz] = useState(false);
  const weekQuiz = quizzes[week.id];

  return (
    <div className={`border rounded-lg overflow-hidden bg-white ${isLocked ? 'opacity-60' : ''}`}>
      <button 
        onClick={() => !isLocked && onExpand()} 
        className={`w-full px-4 py-3 flex items-center gap-3 transition-colors ${isLocked ? 'cursor-not-allowed' : 'hover:bg-gray-50'}`}
      >
        {/* Lock/Unlock/Complete Icon */}
        <div className="flex-shrink-0">
          {isLocked ? (
            <Lock className="w-6 h-6 text-gray-400" />
          ) : completed ? (
            <CheckCircle2 className="w-6 h-6 text-emerald-500" />
          ) : (
            <Unlock className="w-6 h-6 text-emerald-400" />
          )}
        </div>
        
        <div className="flex-1 text-left">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`font-medium ${isLocked ? 'text-gray-400' : completed ? 'text-gray-500' : 'text-gray-800'}`}>
              {week.title}
            </span>
            {isLocked && <Badge variant="locked">🔒 Locked</Badge>}
            {week.isNew && !isLocked && <Badge variant="new">NEW</Badge>}
            {week.isRevised && !isLocked && <Badge variant="revised">REVISED</Badge>}
            {week.hasCapstone && !isLocked && <Badge variant="capstone">★ Capstone Start</Badge>}
            {quizPassed && <Badge variant="passed">Quiz ✓</Badge>}
          </div>
          {isLocked && (
            <p className="text-xs text-gray-400 mt-1">Complete previous weeks to unlock</p>
          )}
        </div>
        
        {!isLocked && (
          expanded ? <ChevronDown className="w-5 h-5 text-gray-400" /> : <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </button>
      
      {expanded && !isLocked && (
        <div className="px-4 pb-4 space-y-4">
          {/* Sessions (for regular weeks) or Topics (for pre-work) */}
          {week.sessions && week.sessions.length > 0 && (
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
          )}
          
          {/* Topics (for pre-work) */}
          {week.topics && week.topics.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                <Target className="w-4 h-4" /> Topics Covered
              </h4>
              <ul className="space-y-1">
                {week.topics.map((topic, i) => (
                  <li key={i} className="text-sm text-gray-600 pl-4 border-l-2 border-emerald-200">{topic}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Async Work */}
          {week.asyncWork && week.asyncWork.length > 0 && (
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
          )}
          
          {/* Resources */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
              <BookOpen className="w-4 h-4" /> Resources
            </h4>
            <div className="grid gap-1">
              {week.resources.map((resource, i) => <ResourceLink key={i} resource={resource} />)}
            </div>
          </div>
          
          {/* Quiz Section */}
          {weekQuiz && (
            <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border border-purple-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${quizPassed ? 'bg-green-100' : 'bg-purple-100'}`}>
                    <HelpCircle className={`w-5 h-5 ${quizPassed ? 'text-green-600' : 'text-purple-600'}`} />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-800">{weekQuiz.title}</h4>
                    <p className="text-sm text-gray-600">
                      {quizPassed 
                        ? `Passed with ${quizResult?.score}%` 
                        : `${weekQuiz.questions.length} questions • ${weekQuiz.passingScore}% to pass`}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setShowQuiz(true)}
                  className={`px-4 py-2 rounded-lg font-medium ${
                    quizPassed 
                      ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                      : 'bg-purple-600 text-white hover:bg-purple-700'
                  }`}
                >
                  {quizPassed ? 'Retake Quiz' : 'Take Quiz'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Quiz Modal */}
      {showQuiz && weekQuiz && (
        <QuizModal
          quiz={weekQuiz}
          weekId={week.id}
          userId={userId}
          previousResult={quizResult}
          onComplete={onQuizComplete}
          onClose={() => setShowQuiz(false)}
        />
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
  const { user, profile, isAdmin, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [completedItems, setCompletedItems] = useState({});
  const [expandedItems, setExpandedItems] = useState({});
  const [activeTab, setActiveTab] = useState('curriculum');
  const [totalTime, setTotalTime] = useState(0);
  const [completedQuizzes, setCompletedQuizzes] = useState({});
  const [quizResults, setQuizResults] = useState({});
  const [dataLoading, setDataLoading] = useState(true);
  const lastLogRef = useRef(Date.now());

  // Load all data from database
  useEffect(() => {
    if (user) {
      loadAllData();
    }
  }, [user]);

  const loadAllData = async () => {
    setDataLoading(true);
    await Promise.all([loadProgress(), loadStats(), loadQuizResults()]);
    setDataLoading(false);
  };

  // Time tracking - log every 5 minutes
  useEffect(() => {
    if (!user) return;
    
    lastLogRef.current = Date.now();
    
    const interval = setInterval(async () => {
      const now = Date.now();
      const secondsSpent = Math.floor((now - lastLogRef.current) / 1000);
      
      if (secondsSpent >= 60 && secondsSpent <= 600) {
        await logTimeSpent(user.id, 'session', secondsSpent);
        lastLogRef.current = now;
        setTotalTime(prev => prev + secondsSpent);
      } else {
        lastLogRef.current = now;
      }
    }, 300000);

    return () => clearInterval(interval);
  }, [user?.id]);

  // Log time on unmount
  useEffect(() => {
    if (!user) return;
    
    const userId = user.id;
    
    return () => {
      const secondsSpent = Math.floor((Date.now() - lastLogRef.current) / 1000);
      if (secondsSpent > 10 && secondsSpent < 600) {
        logTimeSpent(userId, 'session', secondsSpent);
      }
    };
  }, [user?.id]);

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
  };

  const loadQuizResults = async () => {
    const { data } = await getQuizResults(user.id);
    if (data) {
      const quizMap = {};
      const resultMap = {};
      data.forEach(q => {
        if (q.passed) quizMap[q.week_id] = true;
        resultMap[q.week_id] = q;
      });
      setCompletedQuizzes(quizMap);
      setQuizResults(resultMap);
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

  const handleQuizComplete = (quizId, weekId) => {
    setCompletedQuizzes(prev => ({ ...prev, [weekId]: true }));
    loadQuizResults(); // Reload to get the score
  };

  const handleSignOut = async () => {
    const secondsSpent = Math.floor((Date.now() - lastLogRef.current) / 1000);
    if (secondsSpent > 10 && secondsSpent < 600) {
      await logTimeSpent(user.id, 'session', secondsSpent);
    }
    await signOut();
    navigate('/');
  };

  // Calculate progress
  const totalItems = getTotalWeeks();
  const completedCount = Object.values(completedItems).filter(Boolean).length;
  const progressPercent = Math.round((completedCount / totalItems) * 100);
  const quizzesPassedCount = Object.values(completedQuizzes).filter(Boolean).length;

  // Show loading spinner while auth or data is loading
  if (authLoading || dataLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-emerald-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-10 h-10 border-4 border-emerald-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-600">Loading your progress...</p>
        </div>
      </div>
    );
  }

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
                <HelpCircle className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{quizzesPassedCount}/{Object.keys(quizzes).length}</p>
                <p className="text-xs text-gray-500">Quizzes Passed</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-100 rounded-lg">
                <Award className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{Object.keys(completedItems).length}/{getTotalWeeks()}</p>
                <p className="text-xs text-gray-500">Weeks Done</p>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b mb-6">
          {[
            { id: 'curriculum', label: 'Curriculum', icon: BookOpen },
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
            {/* Unlock Progress Info */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Unlock className="w-5 h-5 text-blue-600 mt-0.5" />
                <div>
                  <h4 className="font-medium text-blue-800">Progressive Learning Path</h4>
                  <p className="text-sm text-blue-600">Complete each week's quiz to unlock the next week. Pre-work and Week 1 are unlocked to start.</p>
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
                onQuizComplete={handleQuizComplete}
                isLocked={false}
                quizPassed={completedQuizzes[curriculum.preWork.id]}
                quizResult={quizResults[curriculum.preWork.id]}
              />
            </div>

            {/* Modules */}
            {curriculum.modules.map(module => (
              <div key={module.id}>
                <h2 className="text-lg font-semibold text-gray-800 mb-3">{module.title}</h2>
                <div className="space-y-2">
                  {module.weeks.map(week => {
                    const isLocked = !isWeekUnlocked(week.id, completedQuizzes);
                    return (
                      <WeekCard
                        key={week.id}
                        week={week}
                        completed={completedItems[week.id]}
                        onToggle={() => toggleComplete(week.id)}
                        expanded={expandedItems[week.id]}
                        onExpand={() => toggleExpand(week.id)}
                        userId={user?.id}
                        onQuizComplete={handleQuizComplete}
                        isLocked={isLocked}
                        quizPassed={completedQuizzes[week.id]}
                        quizResult={quizResults[week.id]}
                      />
                    );
                  })}
                </div>
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
        AI Fundamentals Bootcamp v3.1 - Progress saves automatically
      </footer>
    </div>
  );
}

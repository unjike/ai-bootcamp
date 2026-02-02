import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, Users, Clock, TrendingUp, Award, ChevronDown, ChevronRight,
  ArrowLeft, Search, Download, BarChart3, User, Calendar,
  CheckCircle2, XCircle, AlertCircle
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useAuth } from '../lib/AuthContext';
import { getAllStudents } from '../lib/supabase';
import { getTotalWeeks, getAllWeeks } from '../lib/curriculum';

const formatTime = (seconds) => {
  if (!seconds) return '0m';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
};

const formatDate = (dateString) => {
  if (!dateString) return 'Never';
  return new Date(dateString).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const StudentRow = ({ student, onClick, isExpanded }) => {
  const totalWeeks = getTotalWeeks();
  const quizzesPassed = student.quiz_results?.filter(q => q.passed).length || 0;
  const progressPercent = Math.round((quizzesPassed / totalWeeks) * 100);
  const totalTime = student.time_logs?.reduce((sum, log) => sum + log.seconds_spent, 0) || 0;
  
  const getStatusColor = () => {
    if (progressPercent >= 75) return 'text-emerald-600 bg-emerald-100';
    if (progressPercent >= 50) return 'text-blue-600 bg-blue-100';
    if (progressPercent >= 25) return 'text-amber-600 bg-amber-100';
    return 'text-gray-600 bg-gray-100';
  };

  return (
    <tr className="hover:bg-gray-50 cursor-pointer transition-colors" onClick={onClick}>
      <td className="px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-full flex items-center justify-center text-white font-medium text-sm">
            {student.full_name?.charAt(0) || student.email?.charAt(0) || '?'}
          </div>
          <div>
            <p className="font-medium text-gray-800">{student.full_name || 'No name'}</p>
            <p className="text-xs text-gray-500">{student.email}</p>
          </div>
        </div>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className="h-full bg-emerald-500 rounded-full transition-all" style={{ width: `${progressPercent}%` }} />
          </div>
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${getStatusColor()}`}>{progressPercent}%</span>
        </div>
      </td>
      <td className="px-4 py-3 text-sm text-gray-600">{quizzesPassed}/{totalWeeks} weeks</td>
      <td className="px-4 py-3 text-sm text-gray-600">{formatTime(totalTime)}</td>
      <td className="px-4 py-3 text-sm text-gray-600">{quizzesPassed}/{totalWeeks} passed</td>
      <td className="px-4 py-3 text-sm text-gray-500">{formatDate(student.created_at)}</td>
      <td className="px-4 py-3">
        {isExpanded ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
      </td>
    </tr>
  );
};

const StudentDetailPanel = ({ student }) => {
  const allWeeks = getAllWeeks();
  const quizResults = student.quiz_results || [];
  const quizResultsMap = {};
  quizResults.forEach(q => { quizResultsMap[q.week_id] = q; });
  
  const timeByDate = {};
  student.time_logs?.forEach(log => {
    const date = new Date(log.logged_at).toLocaleDateString();
    timeByDate[date] = (timeByDate[date] || 0) + log.seconds_spent;
  });
  
  const activityData = Object.entries(timeByDate).slice(-7).map(([date, seconds]) => ({
    date: new Date(date).toLocaleDateString('en-US', { weekday: 'short' }),
    minutes: Math.round(seconds / 60)
  }));

  return (
    <tr>
      <td colSpan={7} className="px-4 py-4 bg-gray-50 border-t">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Week Progress with Quiz Status */}
          <div>
            <h4 className="font-semibold text-gray-800 mb-3">Week & Quiz Progress</h4>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {allWeeks.map(week => {
                const quizResult = quizResultsMap[week.id];
                const quizPassed = quizResult?.passed;
                return (
                  <div key={week.id} className="flex items-center gap-2 text-sm">
                    {quizPassed ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-gray-300 flex-shrink-0" />
                    )}
                    <span className={quizPassed ? 'text-gray-700' : 'text-gray-400'}>{week.title}</span>
                    {quizResult && (
                      <span className={`ml-auto text-xs px-2 py-0.5 rounded-full ${quizResult.passed ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        Quiz: {quizResult.score}%
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recent Activity Chart */}
          <div>
            <h4 className="font-semibold text-gray-800 mb-3">Recent Activity (Last 7 Days)</h4>
            {activityData.length > 0 ? (
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={activityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(value) => [`${value} min`, 'Time']} />
                  <Bar dataKey="minutes" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-32 flex items-center justify-center text-gray-400 text-sm">No activity data yet</div>
            )}
          </div>

          {/* Quiz Summary */}
          <div>
            <h4 className="font-semibold text-gray-800 mb-3">Quiz Performance</h4>
            {quizResults.length > 0 ? (
              <div className="space-y-2">
                <div className="flex gap-4 text-sm">
                  <div className="bg-green-50 px-3 py-2 rounded-lg">
                    <span className="text-green-700 font-semibold">{quizResults.filter(q => q.passed).length}</span>
                    <span className="text-green-600 ml-1">Passed</span>
                  </div>
                  <div className="bg-red-50 px-3 py-2 rounded-lg">
                    <span className="text-red-700 font-semibold">{quizResults.filter(q => !q.passed).length}</span>
                    <span className="text-red-600 ml-1">Failed</span>
                  </div>
                  <div className="bg-blue-50 px-3 py-2 rounded-lg">
                    <span className="text-blue-700 font-semibold">{Math.round(quizResults.reduce((sum, q) => sum + q.score, 0) / quizResults.length)}%</span>
                    <span className="text-blue-600 ml-1">Avg Score</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-400 text-sm">No quizzes attempted yet</p>
            )}
          </div>
        </div>
      </td>
    </tr>
  );
};

export default function AdminDashboard() {
  const { isAdmin, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedStudent, setExpandedStudent] = useState(null);
  const [sortBy, setSortBy] = useState('progress');

  useEffect(() => {
    // Wait for auth to finish loading before checking admin status
    if (authLoading) return;
    
    if (!isAdmin) {
      navigate('/dashboard');
      return;
    }
    loadStudents();
  }, [isAdmin, authLoading]);

  const loadStudents = async () => {
    setLoading(true);
    const { data } = await getAllStudents();
    if (data) setStudents(data);
    setLoading(false);
  };

  const totalWeeks = getTotalWeeks();

  const filteredStudents = students
    .filter(s => s.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) || s.email?.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => {
      const aProgress = (a.progress?.filter(p => p.completed).length || 0) / totalWeeks;
      const bProgress = (b.progress?.filter(p => p.completed).length || 0) / totalWeeks;
      const aTime = a.time_logs?.reduce((sum, log) => sum + log.seconds_spent, 0) || 0;
      const bTime = b.time_logs?.reduce((sum, log) => sum + log.seconds_spent, 0) || 0;
      
      switch (sortBy) {
        case 'progress': return bProgress - aProgress;
        case 'time': return bTime - aTime;
        case 'name': return (a.full_name || '').localeCompare(b.full_name || '');
        case 'recent': return new Date(b.created_at) - new Date(a.created_at);
        default: return 0;
      }
    });

  const totalStudents = students.length;
  const avgProgress = totalStudents > 0 
    ? Math.round(students.reduce((sum, s) => sum + ((s.progress?.filter(p => p.completed).length || 0) / totalWeeks) * 100, 0) / totalStudents)
    : 0;
  const totalTimeAll = students.reduce((sum, s) => sum + (s.time_logs?.reduce((t, log) => t + log.seconds_spent, 0) || 0), 0);
  const avgTime = totalStudents > 0 ? Math.round(totalTimeAll / totalStudents) : 0;
  const activeStudents = students.filter(s => {
    const lastWeek = Date.now() - 7 * 24 * 60 * 60 * 1000;
    return s.time_logs?.some(log => new Date(log.logged_at) > lastWeek);
  }).length;

  const progressDistribution = [
    { name: '0-25%', value: students.filter(s => { const p = (s.quiz_results?.filter(q => q.passed).length || 0) / totalWeeks * 100; return p >= 0 && p < 25; }).length, color: '#ef4444' },
    { name: '25-50%', value: students.filter(s => { const p = (s.quiz_results?.filter(q => q.passed).length || 0) / totalWeeks * 100; return p >= 25 && p < 50; }).length, color: '#f59e0b' },
    { name: '50-75%', value: students.filter(s => { const p = (s.quiz_results?.filter(q => q.passed).length || 0) / totalWeeks * 100; return p >= 50 && p < 75; }).length, color: '#3b82f6' },
    { name: '75-100%', value: students.filter(s => { const p = (s.quiz_results?.filter(q => q.passed).length || 0) / totalWeeks * 100; return p >= 75; }).length, color: '#10b981' },
  ].filter(d => d.value > 0);

  const exportCSV = () => {
    const headers = ['Name', 'Email', 'Progress %', 'Quizzes Passed', 'Time Spent', 'Joined'];
    const rows = students.map(s => {
      const time = s.time_logs?.reduce((sum, log) => sum + log.seconds_spent, 0) || 0;
      const quizzesPassed = s.quiz_results?.filter(q => q.passed).length || 0;
      return [s.full_name || '', s.email, Math.round((quizzesPassed / totalWeeks) * 100), quizzesPassed, formatTime(time), formatDate(s.created_at)];
    });
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bootcamp-students-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  if (loading || authLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button onClick={() => navigate('/dashboard')} className="p-2 hover:bg-gray-100 rounded-lg">
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-800">Admin Dashboard</h1>
                  <p className="text-sm text-gray-500">Student Performance Tracking</p>
                </div>
              </div>
            </div>
            <button onClick={exportCSV} className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600">
              <Download className="w-4 h-4" /> Export CSV
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg"><Users className="w-5 h-5 text-blue-600" /></div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{totalStudents}</p>
                <p className="text-xs text-gray-500">Total Students</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-100 rounded-lg"><TrendingUp className="w-5 h-5 text-emerald-600" /></div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{avgProgress}%</p>
                <p className="text-xs text-gray-500">Avg Progress</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg"><Clock className="w-5 h-5 text-purple-600" /></div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{formatTime(avgTime)}</p>
                <p className="text-xs text-gray-500">Avg Time/Student</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-100 rounded-lg"><Award className="w-5 h-5 text-amber-600" /></div>
              <div>
                <p className="text-2xl font-bold text-gray-800">{activeStudents}</p>
                <p className="text-xs text-gray-500">Active This Week</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-4">Progress Distribution</h3>
            {progressDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={progressDistribution} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                    {progressDistribution.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.color} />))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-48 flex items-center justify-center text-gray-400">No student data yet</div>
            )}
          </div>

          <div className="bg-white rounded-xl p-4 border shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-4">Quiz Completion by Week</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={getAllWeeks().map(week => ({
                name: week.id.replace('week-', 'W').replace('pre-work', 'Pre'),
                completed: students.filter(s => s.quiz_results?.some(q => q.week_id === week.id && q.passed)).length
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="completed" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
          <div className="p-4 border-b flex flex-col sm:flex-row gap-4 justify-between">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input type="text" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} placeholder="Search students..."
                className="pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none w-full sm:w-64" />
            </div>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 outline-none">
              <option value="progress">Sort by Progress</option>
              <option value="time">Sort by Time Spent</option>
              <option value="name">Sort by Name</option>
              <option value="recent">Sort by Join Date</option>
            </select>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Student</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Progress</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Completed</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Time Spent</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Quizzes</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Joined</th>
                  <th className="px-4 py-3 w-8"></th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {filteredStudents.length > 0 ? filteredStudents.map(student => (
                  <React.Fragment key={student.id}>
                    <StudentRow student={student} onClick={() => setExpandedStudent(expandedStudent === student.id ? null : student.id)} isExpanded={expandedStudent === student.id} />
                    {expandedStudent === student.id && <StudentDetailPanel student={student} />}
                  </React.Fragment>
                )) : (
                  <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-500">{searchTerm ? 'No students found' : 'No students enrolled yet'}</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

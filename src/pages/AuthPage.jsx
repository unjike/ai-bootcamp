import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  Brain, Mail, Lock, User, ArrowRight, Loader2, ArrowLeft,
  CheckCircle2, Calendar, Briefcase, TrendingUp, Sparkles, Code
} from 'lucide-react';
import { signIn, signUp, signInWithGoogle } from '../lib/supabase';

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');

    try {
      if (isLogin) {
        const { data, error } = await signIn(email, password);
        if (error) throw error;
        navigate('/dashboard');
      } else {
        const { data, error } = await signUp(email, password, fullName);
        if (error) throw error;
        setMessage('Check your email for the confirmation link!');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError('');
    try {
      const { error } = await signInWithGoogle();
      if (error) throw error;
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const features = [
    { icon: Calendar, text: '12-week structured curriculum' },
    { icon: Briefcase, text: '8 portfolio projects + capstone' },
    { icon: Code, text: 'Hands-on coding exercises' },
    { icon: Sparkles, text: 'LLMs, RAG & Prompt Engineering' },
    { icon: TrendingUp, text: 'Job-aligned skills for 2026' },
  ];

  const stats = [
    { value: '12', label: 'Weeks' },
    { value: '8+', label: 'Projects' },
    { value: '$88K+', label: 'Avg Salary' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-emerald-50 flex">
      {/* Left Side - Features (hidden on mobile) */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-emerald-600 to-teal-700 p-12 flex-col justify-between">
        <div>
          {/* Back to Home */}
          <Link to="/" className="inline-flex items-center gap-2 text-emerald-100 hover:text-white mb-12 transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">Back to Home</span>
          </Link>

          {/* Logo & Title */}
          <div className="flex items-center gap-4 mb-8">
            <div className="w-14 h-14 bg-white/20 backdrop-blur rounded-xl flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">AI Fundamentals</h1>
              <p className="text-emerald-100">Bootcamp v3.0</p>
            </div>
          </div>

          {/* Headline */}
          <h2 className="text-3xl font-bold text-white mb-4 leading-tight">
            Transform Your Career in Data Science & AI
          </h2>
          <p className="text-emerald-100 text-lg mb-10">
            Go from engineer to job-ready Data Scientist in 12 weeks with hands-on projects and cutting-edge skills.
          </p>

          {/* Features List */}
          <div className="space-y-4">
            {features.map((feature, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-10 h-10 bg-white/20 backdrop-blur rounded-lg flex items-center justify-center">
                  <feature.icon className="w-5 h-5 text-white" />
                </div>
                <span className="text-white font-medium">{feature.text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Stats */}
        <div className="flex gap-8 pt-8 border-t border-white/20">
          {stats.map((stat, i) => (
            <div key={i}>
              <p className="text-3xl font-bold text-white">{stat.value}</p>
              <p className="text-emerald-100 text-sm">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Right Side - Auth Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 lg:p-12">
        <div className="w-full max-w-md">
          {/* Mobile: Back to Home */}
          <Link to="/" className="lg:hidden inline-flex items-center gap-2 text-gray-500 hover:text-emerald-600 mb-6 transition-colors">
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">Back to Home</span>
          </Link>

          {/* Mobile: Logo */}
          <div className="lg:hidden text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl mb-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-800">AI Fundamentals Bootcamp</h1>
            <p className="text-gray-500 mt-1">12-Week Curriculum • v3.0</p>
          </div>

          {/* Auth Card */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              {isLogin ? 'Welcome back!' : 'Create your account'}
            </h2>
            <p className="text-gray-500 text-sm mb-6">
              {isLogin ? 'Sign in to continue your learning journey' : 'Start your journey to becoming a Data Scientist'}
            </p>

            {/* Google Sign In */}
            <button
              onClick={handleGoogleSignIn}
              disabled={loading}
              className="w-full flex items-center justify-center gap-3 px-4 py-3 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-gray-300 transition-colors mb-6"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              <span className="text-gray-700 font-medium">Continue with Google</span>
            </button>

            <div className="relative mb-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500">or continue with email</span>
              </div>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-all"
                      placeholder="John Doe"
                      required={!isLogin}
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-all"
                    placeholder="you@example.com"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none transition-all"
                    placeholder="••••••••"
                    required
                    minLength={6}
                  />
                </div>
              </div>

              {error && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                  {error}
                </div>
              )}

              {message && (
                <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-600 text-sm">
                  {message}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white font-medium rounded-lg hover:from-emerald-600 hover:to-teal-700 transition-all disabled:opacity-50 shadow-lg shadow-emerald-500/25"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    {isLogin ? 'Sign In' : 'Create Account'}
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </form>

            <p className="mt-6 text-center text-sm text-gray-500">
              {isLogin ? "Don't have an account?" : 'Already have an account?'}
              <button
                onClick={() => { setIsLogin(!isLogin); setError(''); setMessage(''); }}
                className="ml-1 text-emerald-600 font-medium hover:text-emerald-700"
              >
                {isLogin ? 'Sign up' : 'Sign in'}
              </button>
            </p>
          </div>

          {/* Target roles */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-500 mb-3">Prepare for entry-level roles:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {['Data Scientist', 'ML Engineer', 'Data Analyst', 'AI Specialist'].map((role, i) => (
                <span key={i} className="px-3 py-1 bg-white border border-gray-200 rounded-full text-xs font-medium text-gray-600">
                  {role}
                </span>
              ))}
            </div>
          </div>

          {/* Trust indicators */}
          <div className="mt-8 flex items-center justify-center gap-6 text-gray-400 text-xs">
            <div className="flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" />
              <span>Free to start</span>
            </div>
            <div className="flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" />
              <span>No credit card</span>
            </div>
            <div className="flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" />
              <span>Cancel anytime</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

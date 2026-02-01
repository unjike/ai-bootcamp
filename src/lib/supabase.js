import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Debug: Log if env vars are missing
if (!supabaseUrl || !supabaseAnonKey) {
  console.error('❌ Missing Supabase environment variables!');
  console.error('VITE_SUPABASE_URL:', supabaseUrl ? '✓ SET' : '✗ MISSING');
  console.error('VITE_SUPABASE_ANON_KEY:', supabaseAnonKey ? '✓ SET' : '✗ MISSING');
}

export const supabase = createClient(
  supabaseUrl || 'https://placeholder.supabase.co',
  supabaseAnonKey || 'placeholder-key'
);

// Auth helpers
export const signUp = async (email, password, fullName) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        full_name: fullName,
        role: 'student'
      }
    }
  });
  return { data, error };
};

export const signIn = async (email, password) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  });
  return { data, error };
};

export const signInWithGoogle = async () => {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: `${window.location.origin}/dashboard`
    }
  });
  return { data, error };
};

export const signOut = async () => {
  const { error } = await supabase.auth.signOut();
  return { error };
};

export const getCurrentUser = async () => {
  const { data: { user } } = await supabase.auth.getUser();
  return user;
};

// Database helpers
export const getUserProfile = async (userId) => {
  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single();
  return { data, error };
};

export const updateUserProfile = async (userId, updates) => {
  const { data, error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', userId);
  return { data, error };
};

// Progress tracking
export const getProgress = async (userId) => {
  const { data, error } = await supabase
    .from('progress')
    .select('*')
    .eq('user_id', userId);
  return { data, error };
};

export const updateProgress = async (userId, weekId, progressData) => {
  const { data, error } = await supabase
    .from('progress')
    .upsert({
      user_id: userId,
      week_id: weekId,
      ...progressData,
      updated_at: new Date().toISOString()
    }, {
      onConflict: 'user_id,week_id'
    });
  return { data, error };
};

// Exercise attempts
export const saveExerciseAttempt = async (userId, exerciseId, code, isCorrect) => {
  const { data, error } = await supabase
    .from('exercise_attempts')
    .insert({
      user_id: userId,
      exercise_id: exerciseId,
      code_submitted: code,
      is_correct: isCorrect,
      submitted_at: new Date().toISOString()
    });
  return { data, error };
};

export const getExerciseAttempts = async (userId, exerciseId = null) => {
  let query = supabase
    .from('exercise_attempts')
    .select('*')
    .eq('user_id', userId)
    .order('submitted_at', { ascending: false });
  
  if (exerciseId) {
    query = query.eq('exercise_id', exerciseId);
  }
  
  const { data, error } = await query;
  return { data, error };
};

// Time tracking
export const logTimeSpent = async (userId, weekId, seconds) => {
  const { data, error } = await supabase
    .from('time_logs')
    .insert({
      user_id: userId,
      week_id: weekId,
      seconds_spent: seconds,
      logged_at: new Date().toISOString()
    });
  return { data, error };
};

export const getTotalTimeSpent = async (userId) => {
  const { data, error } = await supabase
    .from('time_logs')
    .select('seconds_spent')
    .eq('user_id', userId);
  
  if (error) return { data: 0, error };
  
  const total = data.reduce((sum, log) => sum + log.seconds_spent, 0);
  return { data: total, error: null };
};

// Admin functions
export const getAllStudents = async () => {
  const { data, error } = await supabase
    .from('profiles')
    .select(`
      *,
      progress (week_id, completed, completed_at),
      time_logs (seconds_spent, logged_at),
      exercise_attempts (exercise_id, is_correct),
      quiz_results (quiz_id, week_id, score, passed, completed_at)
    `)
    .eq('role', 'student')
    .order('created_at', { ascending: false });
  return { data, error };
};

export const getStudentDetails = async (studentId) => {
  const { data, error } = await supabase
    .from('profiles')
    .select(`
      *,
      progress (*),
      time_logs (*),
      exercise_attempts (*)
    `)
    .eq('id', studentId)
    .single();
  return { data, error };
};

export const isAdmin = async (userId) => {
  const { data, error } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .single();
  
  if (error) return false;
  return data?.role === 'admin';
};

// Quiz functions
export const saveQuizResult = async (userId, quizId, weekId, score, passed, answers) => {
  const { data, error } = await supabase
    .from('quiz_results')
    .upsert({
      user_id: userId,
      quiz_id: quizId,
      week_id: weekId,
      score: score,
      passed: passed,
      answers: answers,
      completed_at: new Date().toISOString()
    }, {
      onConflict: 'user_id,quiz_id'
    });
  return { data, error };
};

export const getQuizResults = async (userId) => {
  try {
    const { data, error } = await supabase
      .from('quiz_results')
      .select('*')
      .eq('user_id', userId);
    return { data: data || [], error };
  } catch (err) {
    console.error('getQuizResults exception:', err);
    return { data: [], error: err };
  }
};

export const getQuizResult = async (userId, quizId) => {
  try {
    const { data, error } = await supabase
      .from('quiz_results')
      .select('*')
      .eq('user_id', userId)
      .eq('quiz_id', quizId)
      .single();
    return { data, error };
  } catch (err) {
    return { data: null, error: err };
  }
};

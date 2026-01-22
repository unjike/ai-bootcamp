import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase, getUserProfile, isAdmin } from './supabase';

const AuthContext = createContext({});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [isAdminUser, setIsAdminUser] = useState(false);
  const [loading, setLoading] = useState(true);
  const [initialized, setInitialized] = useState(false);

  const loadProfile = async (userId) => {
    try {
      console.log('Loading profile for:', userId);
      const { data, error } = await getUserProfile(userId);
      if (error) {
        console.error('Error loading profile:', error);
      }
      setProfile(data);
      
      const adminStatus = await isAdmin(userId);
      console.log('Admin status:', adminStatus);
      setIsAdminUser(adminStatus);
    } catch (error) {
      console.error('loadProfile exception:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Set up auth state listener FIRST
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        console.log('Auth state changed:', event, session?.user?.email);
        
        setUser(session?.user ?? null);
        
        if (session?.user) {
          // Use setTimeout to avoid Supabase deadlock issue
          setTimeout(() => {
            loadProfile(session.user.id);
          }, 0);
        } else {
          setProfile(null);
          setIsAdminUser(false);
          setLoading(false);
        }
        
        setInitialized(true);
      }
    );

    // Then check for existing session
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (error) {
        console.error('getSession error:', error);
        setLoading(false);
        setInitialized(true);
        return;
      }
      
      // Only set if not already initialized by onAuthStateChange
      if (!initialized) {
        setUser(session?.user ?? null);
        if (session?.user) {
          loadProfile(session.user.id);
        } else {
          setLoading(false);
        }
        setInitialized(true);
      }
    }).catch(err => {
      console.error('getSession exception:', err);
      setLoading(false);
      setInitialized(true);
    });

    return () => subscription.unsubscribe();
  }, []);

  const value = {
    user,
    profile,
    isAdmin: isAdminUser,
    loading,
    refreshProfile: () => user && loadProfile(user.id)
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

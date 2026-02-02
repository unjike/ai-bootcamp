import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Brain, ChevronRight, CheckCircle2, Clock, Users, Award, 
  Code, Database, BarChart3, Cpu, MessageSquare, Sparkles,
  BookOpen, Target, TrendingUp, DollarSign, Calendar, ArrowRight,
  ChevronDown, Star, Briefcase, GraduationCap, Zap, MessageCircle
} from 'lucide-react';

// WhatsApp URL
const WHATSAPP_URL = "https://api.whatsapp.com/send/?phone=13017682721&text=Hello%2C+tell+me+more+about+your+AI+bootcamp";

// Skill Badge Component
const SkillBadge = ({ icon: Icon, label, color }) => (
  <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${color} transition-transform hover:scale-105`}>
    <Icon className="w-4 h-4" />
    <span className="text-sm font-medium">{label}</span>
  </div>
);

// Stat Card Component
const StatCard = ({ value, label, icon: Icon }) => (
  <div className="text-center p-6 bg-white rounded-xl shadow-sm border border-gray-100">
    <div className="inline-flex items-center justify-center w-12 h-12 bg-emerald-100 rounded-xl mb-3">
      <Icon className="w-6 h-6 text-emerald-600" />
    </div>
    <p className="text-3xl font-bold text-gray-800">{value}</p>
    <p className="text-sm text-gray-500 mt-1">{label}</p>
  </div>
);

// Week Preview Component
const WeekPreview = ({ week, title, description, isNew }) => (
  <div className="flex gap-4 p-4 rounded-lg hover:bg-gray-50 transition-colors">
    <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center text-white font-bold">
      {week}
    </div>
    <div>
      <div className="flex items-center gap-2">
        <h4 className="font-semibold text-gray-800">{title}</h4>
        {isNew && <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-medium rounded-full">NEW</span>}
      </div>
      <p className="text-sm text-gray-500 mt-1">{description}</p>
    </div>
  </div>
);

// Project Card Component
const ProjectCard = ({ title, skills, impact }) => (
  <div className="bg-white p-5 rounded-xl border border-gray-200 hover:border-emerald-300 hover:shadow-md transition-all">
    <h4 className="font-semibold text-gray-800 mb-2">{title}</h4>
    <p className="text-xs text-emerald-600 font-medium mb-3">{impact}</p>
    <div className="flex flex-wrap gap-1">
      {skills.map((skill, i) => (
        <span key={i} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">{skill}</span>
      ))}
    </div>
  </div>
);

// FAQ Item Component
const FAQItem = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="border-b border-gray-200">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full py-4 flex items-center justify-between text-left hover:text-emerald-600 transition-colors"
      >
        <span className="font-medium text-gray-800">{question}</span>
        <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      {isOpen && (
        <div className="pb-4 text-gray-600 text-sm leading-relaxed">
          {answer}
        </div>
      )}
    </div>
  );
};

export default function LandingPage() {
  const navigate = useNavigate();

  const skills = [
    { icon: Code, label: 'Python', color: 'bg-blue-100 text-blue-700' },
    { icon: Database, label: 'SQL', color: 'bg-purple-100 text-purple-700' },
    { icon: BarChart3, label: 'Machine Learning', color: 'bg-emerald-100 text-emerald-700' },
    { icon: Cpu, label: 'Deep Learning', color: 'bg-orange-100 text-orange-700' },
    { icon: MessageSquare, label: 'LLMs & RAG', color: 'bg-pink-100 text-pink-700' },
    { icon: Sparkles, label: 'Prompt Engineering', color: 'bg-indigo-100 text-indigo-700' },
  ];

  const curriculum = [
    { week: 'Pre', title: 'AI Ethics & Responsible AI', description: 'Bias detection, fairness, privacy, governance frameworks', isNew: true },
    { week: '1-3', title: 'Foundations', description: 'What is AI, Data Thinking, Python & SQL fundamentals' },
    { week: '4-6', title: 'Core Machine Learning', description: 'Regression, Classification, Model Evaluation & Deployment' },
    { week: '7', title: 'Recommendations', description: 'Collaborative filtering, content-based, matrix factorization', isNew: true },
    { week: '8-9', title: 'Deep Learning', description: 'Neural networks, CNNs, Transfer Learning, Transformers' },
    { week: '10-11', title: 'Generative AI & LLMs', description: 'Prompt engineering, LLM APIs, RAG, Agentic AI', isNew: true },
    { week: '12', title: 'Capstone & Career', description: 'Project presentations, portfolio review, interview prep' },
  ];

  const projects = [
    { title: 'IBM HR Attrition Analysis', skills: ['EDA', 'Visualization', 'Statistics'], impact: '$15K per employee saved' },
    { title: 'TechMart SQL Pipeline', skills: ['SQL', 'CTEs', 'Window Functions'], impact: '$2.4M transaction analysis' },
    { title: 'Lending Club Prediction', skills: ['Regression', 'Classification', 'Deployment'], impact: '$9K per prevented default' },
    { title: 'MovieLens Recommender', skills: ['Collaborative Filtering', 'SVD'], impact: 'Netflix-style recommendations' },
    { title: 'Heart Disease Prediction', skills: ['Neural Networks', 'Keras'], impact: '$75K per diagnosis value' },
    { title: 'LLM Customer Support', skills: ['Prompt Engineering', 'APIs'], impact: '$500K+ annual savings' },
  ];

  const faqs = [
    { 
      question: 'Who is this bootcamp for?', 
      answer: 'This bootcamp is designed for professionals with a bachelor\'s degree in engineering, computer science, mathematics, physics, or related quantitative fields who want to transition into data science and AI roles. Basic programming familiarity is helpful but not required.' 
    },
    { 
      question: 'How much time do I need to commit?', 
      answer: 'Plan for 13-16 hours per week: 3 hours of live sessions (2x 90-minute sessions), 4-8 hours of async work, and additional project/capstone time in later weeks. The program is designed for working professionals.' 
    },
    { 
      question: 'What will I be able to do after completing the bootcamp?', 
      answer: 'You\'ll be prepared for Data Scientist, Data Analyst, ML Engineer, and AI Specialist entry-level roles. You\'ll have 8 portfolio projects plus a capstone on GitHub, practical experience with industry tools, and interview-ready skills.' 
    },
    { 
      question: 'Do I need prior machine learning experience?', 
      answer: 'No! We start from fundamentals. If you have a quantitative background and basic coding exposure, you\'ll be able to follow along. The curriculum builds progressively from basics to advanced topics.' 
    },
    { 
      question: 'What makes this different from other bootcamps?', 
      answer: 'Our curriculum is aligned with 2025-2026 job market requirements, includes cutting-edge topics like RAG and Agentic AI, emphasizes business impact quantification, and provides 11 real-world portfolio projects with actual datasets—not toy examples.' 
    },
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 bg-white/80 backdrop-blur-md z-50 border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="font-bold text-gray-800">AI Fundamentals Bootcamp</span>
          </div>
          <div className="flex items-center gap-4">
            <a href="#curriculum" className="hidden sm:block text-sm text-gray-600 hover:text-emerald-600">Curriculum</a>
            <a href="#projects" className="hidden sm:block text-sm text-gray-600 hover:text-emerald-600">Projects</a>
            <a href="#faq" className="hidden sm:block text-sm text-gray-600 hover:text-emerald-600">FAQ</a>
            <button 
              onClick={() => navigate('/login')}
              className="px-4 py-2 bg-emerald-500 text-white text-sm font-medium rounded-lg hover:bg-emerald-600 transition-colors"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4" />
              <span>Version 3.0 — Updated for 2026 Job Market</span>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight mb-6">
              Become a <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 to-teal-600">Data Scientist</span> in 12 Weeks
            </h1>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Transform your engineering background into a high-paying AI career. 
              Learn Python, ML, Deep Learning, and cutting-edge LLM skills with 
              hands-on projects that impress employers.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button 
                onClick={() => navigate('/login')}
                className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-teal-600 text-white font-semibold rounded-xl hover:from-emerald-600 hover:to-teal-700 transition-all shadow-lg shadow-emerald-500/25 flex items-center justify-center gap-2"
              >
                Start Learning Free <ArrowRight className="w-5 h-5" />
              </button>
              <a 
                href="#curriculum"
                className="px-8 py-4 bg-white text-gray-700 font-semibold rounded-xl border border-gray-200 hover:border-emerald-300 hover:text-emerald-600 transition-all flex items-center justify-center gap-2"
              >
                View Curriculum <ChevronDown className="w-5 h-5" />
              </a>
            </div>
            <div className="mt-4">
              <a 
                href={WHATSAPP_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-gray-600 hover:text-emerald-600 transition-colors"
              >
                <MessageCircle className="w-5 h-5" />
                <span>Questions? Chat with us on WhatsApp</span>
              </a>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-16">
            <StatCard value="12" label="Weeks" icon={Calendar} />
            <StatCard value="8+" label="Portfolio Projects" icon={Briefcase} />
            <StatCard value="$88K+" label="Median Salary" icon={DollarSign} />
            <StatCard value="100%" label="Job-Aligned" icon={Target} />
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Skills You'll Master</h2>
            <p className="text-gray-600">Industry-demanded skills aligned with 2025-2026 job postings</p>
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            {skills.map((skill, i) => (
              <SkillBadge key={i} {...skill} />
            ))}
            <SkillBadge icon={Database} label="TensorFlow/Keras" color="bg-red-100 text-red-700" />
            <SkillBadge icon={BarChart3} label="Data Visualization" color="bg-yellow-100 text-yellow-700" />
            <SkillBadge icon={Cpu} label="Neural Networks" color="bg-cyan-100 text-cyan-700" />
            <SkillBadge icon={Database} label="Vector DBs (ChromaDB/Pinecone)" color="bg-violet-100 text-violet-700" />
            <SkillBadge icon={BookOpen} label="AI Ethics" color="bg-gray-100 text-gray-700" />
            <SkillBadge icon={Code} label="Git/GitHub" color="bg-gray-800 text-white" />
            <SkillBadge icon={Zap} label="Streamlit" color="bg-rose-100 text-rose-700" />
          </div>
        </div>
      </section>

      {/* Target Roles */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Launch Your AI Career</h2>
            <p className="text-gray-600">Graduates are prepared for these high-demand roles</p>
          </div>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { title: 'Data Scientist', salary: '$88K - $110K', icon: BarChart3 },
              { title: 'Data Analyst', salary: '$57K - $75K', icon: TrendingUp },
              { title: 'ML Engineer', salary: '$90K - $103K', icon: Cpu },
              { title: 'AI Specialist', salary: '$70K - $102K', icon: Brain },
            ].map((role, i) => (
              <div key={i} className="bg-white p-6 rounded-xl border border-gray-200 hover:border-emerald-300 hover:shadow-lg transition-all text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl mb-4">
                  <role.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-semibold text-gray-800 mb-1">{role.title}</h3>
                <p className="text-emerald-600 font-medium">{role.salary}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Curriculum Section */}
      <section id="curriculum" className="py-16 px-4 bg-gray-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">12-Week Curriculum</h2>
            <p className="text-gray-600">Comprehensive, hands-on learning path from fundamentals to advanced AI</p>
          </div>
          <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden">
            {curriculum.map((item, i) => (
              <WeekPreview key={i} {...item} />
            ))}
          </div>
          <div className="mt-8 text-center">
            <div className="inline-flex items-center gap-6 text-sm text-gray-500">
              <span className="flex items-center gap-2"><Clock className="w-4 h-4" /> 3 hrs live/week</span>
              <span className="flex items-center gap-2"><BookOpen className="w-4 h-4" /> 4-8 hrs async/week</span>
              <span className="flex items-center gap-2"><Code className="w-4 h-4" /> Hands-on focused</span>
            </div>
          </div>
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Build a Killer Portfolio</h2>
            <p className="text-gray-600">8 real-world projects + a capstone with quantified business impact</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project, i) => (
              <ProjectCard key={i} {...project} />
            ))}
          </div>
          <div className="mt-8 text-center">
            <p className="text-gray-500 text-sm">+ Image Classification, Document Q&A with RAG, and your personalized Capstone Project</p>
          </div>
        </div>
      </section>

      {/* Why This Bootcamp */}
      <section className="py-16 px-4 bg-gradient-to-br from-emerald-500 to-teal-600 text-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold mb-4">Why Choose This Bootcamp?</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { icon: Target, title: 'Job Market Aligned', description: 'Curriculum built from analyzing 2025-2026 job postings. Every skill you learn is in demand.' },
              { icon: Sparkles, title: 'Cutting-Edge Content', description: 'Includes LLMs, RAG, Agentic AI, and prompt engineering—skills most bootcamps don\'t cover yet.' },
              { icon: DollarSign, title: 'Business Impact Focus', description: 'Every project quantifies business value. Speak the language employers want to hear.' },
              { icon: GraduationCap, title: 'AI Ethics Foundation', description: 'Start with responsible AI principles. Stand out as a thoughtful, mature candidate.' },
              { icon: Briefcase, title: 'Portfolio-First', description: '8 GitHub-ready projects + capstone prove your skills better than any certificate.' },
              { icon: Award, title: 'Interview Ready', description: 'Embedded interview prep, STAR framework practice, and resume bullet suggestions.' },
            ].map((item, i) => (
              <div key={i} className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 bg-white/20 rounded-xl mb-4">
                  <item.icon className="w-6 h-6" />
                </div>
                <h3 className="font-semibold text-lg mb-2">{item.title}</h3>
                <p className="text-emerald-100 text-sm">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Program Structure */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-emerald-600">1</span>
              </div>
              <h3 className="font-semibold text-gray-800 mb-2">Complete Pre-Work</h3>
              <p className="text-gray-500 text-sm">Start with AI Ethics module (~8 hours self-paced) to build your foundation before week 1.</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-emerald-600">2</span>
              </div>
              <h3 className="font-semibold text-gray-800 mb-2">Learn & Build Weekly</h3>
              <p className="text-gray-500 text-sm">Attend live sessions, complete exercises, pass quizzes, and build portfolio projects each week.</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-emerald-600">3</span>
              </div>
              <h3 className="font-semibold text-gray-800 mb-2">Present & Launch</h3>
              <p className="text-gray-500 text-sm">Complete your capstone, polish your GitHub portfolio, and start applying to your dream roles.</p>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-16 px-4 bg-gray-50">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Frequently Asked Questions</h2>
          </div>
          <div className="bg-white rounded-2xl border border-gray-200 p-6">
            {faqs.map((faq, i) => (
              <FAQItem key={i} {...faq} />
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-20 px-4 bg-gradient-to-br from-gray-900 to-gray-800 text-white">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">Ready to Transform Your Career?</h2>
          <p className="text-gray-300 mb-8 text-lg">
            Join the next generation of data scientists and AI engineers. 
            Your future in AI starts today.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={() => navigate('/login')}
              className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-teal-600 text-white font-semibold rounded-xl hover:from-emerald-600 hover:to-teal-700 transition-all shadow-lg shadow-emerald-500/25 flex items-center justify-center gap-2"
            >
              Get Started Now <ArrowRight className="w-5 h-5" />
            </button>
            <a 
              href={WHATSAPP_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 bg-transparent text-white font-semibold rounded-xl border border-gray-600 hover:border-emerald-500 hover:text-emerald-400 transition-all flex items-center justify-center gap-2"
            >
              <MessageCircle className="w-5 h-5" />
              Chat on WhatsApp
            </a>
          </div>
          <p className="mt-6 text-gray-400 text-sm">Free to start • No credit card required</p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 bg-gray-900 text-gray-400 text-sm">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-emerald-500" />
            <span>AI Fundamentals Bootcamp</span>
          </div>
          <p>© 2026 AI Fundamentals Bootcamp. Curriculum v3.0</p>
        </div>
      </footer>
    </div>
  );
}

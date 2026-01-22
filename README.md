# AI Fundamentals Bootcamp App

A complete learning management system for your 12-week AI/ML bootcamp with user authentication, progress tracking, time analytics, and admin dashboard.

## Features

### For Students
- 📚 **Curriculum Tracker** - Track progress through all 12 weeks + pre-work
- 💻 **Interactive Exercises** - 9 hands-on coding exercises with solutions
- 🔗 **Resource Library** - Curated links to courses, docs, and tutorials
- 📊 **Personal Dashboard** - View progress, time spent, exercises completed
- 💾 **Cloud Sync** - Progress syncs across devices

### For Admins
- 👥 **Student Overview** - See all enrolled students at a glance
- 📈 **Performance Analytics** - Progress distribution, completion rates
- ⏱️ **Engagement Tracking** - Time spent per student, active users
- 📋 **Detailed Reports** - Expand any student to see week-by-week progress
- 📥 **Export to CSV** - Download student data for reporting

---

## 🚀 Complete Setup Guide

### Step 1: Create Supabase Project (5 minutes)

1. Go to [supabase.com](https://supabase.com) and sign up (free)
2. Click **"New Project"**
3. Choose your organization, name it `ai-bootcamp`, set a password
4. Wait ~2 minutes for provisioning

### Step 2: Set Up Database (3 minutes)

1. In your Supabase dashboard, go to **SQL Editor**
2. Click **"New Query"**
3. Copy the entire contents of `supabase-schema.sql` from this project
4. Paste it and click **"Run"**
5. You should see "Success" messages

### Step 3: Enable Google Auth (Optional, 5 minutes)

1. In Supabase, go to **Authentication** → **Providers**
2. Find **Google** and enable it
3. Go to [Google Cloud Console](https://console.cloud.google.com/)
4. Create a new project or select existing
5. Go to **APIs & Services** → **Credentials**
6. Create **OAuth 2.0 Client ID** (Web application)
7. Add authorized redirect URI: `https://YOUR-PROJECT-ID.supabase.co/auth/v1/callback`
8. Copy Client ID and Client Secret back to Supabase Google provider settings

### Step 4: Get Your API Keys

1. In Supabase, go to **Settings** → **API**
2. Copy:
   - **Project URL** (looks like `https://xxxxx.supabase.co`)
   - **anon/public key** (long string starting with `eyJ...`)

### Step 5: Configure Environment

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your Supabase credentials:
   ```
   VITE_SUPABASE_URL=https://your-project-id.supabase.co
   VITE_SUPABASE_ANON_KEY=your-anon-key-here
   ```

### Step 6: Deploy to Vercel (3 minutes)

1. Push code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/ai-bootcamp.git
   git push -u origin main
   ```

2. Go to [vercel.com](https://vercel.com) and sign in with GitHub

3. Click **"Add New Project"** → Select your repo

4. **Add Environment Variables** before deploying:
   - `VITE_SUPABASE_URL` = your Supabase URL
   - `VITE_SUPABASE_ANON_KEY` = your Supabase anon key

5. Click **Deploy**!

### Step 7: Make Yourself Admin

1. Sign up for an account on your deployed app
2. Go to Supabase → **SQL Editor**
3. Run:
   ```sql
   UPDATE profiles 
   SET role = 'admin' 
   WHERE email = 'your-email@example.com';
   ```
4. Refresh your app - you'll now see the Admin Dashboard button!

---

## 📁 Project Structure

```
ai-bootcamp/
├── src/
│   ├── components/          # Reusable UI components
│   ├── lib/
│   │   ├── supabase.js      # Supabase client & helpers
│   │   ├── AuthContext.jsx  # Auth state management
│   │   └── curriculum.js    # Curriculum data
│   ├── pages/
│   │   ├── AuthPage.jsx     # Login/Signup
│   │   ├── StudentDashboard.jsx
│   │   └── AdminDashboard.jsx
│   ├── App.jsx              # Routes
│   ├── main.jsx             # Entry point
│   └── index.css            # Styles
├── supabase-schema.sql      # Database setup script
├── .env.example             # Environment template
└── README.md
```

---

## 🔧 Local Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

---

## ✏️ Customizing the Curriculum

Edit `src/lib/curriculum.js` to modify:

### Add a New Week
```javascript
{
  id: 'week-13',
  title: 'Week 13: Advanced Topics',
  sessions: [
    'Session 1: Topic A',
    'Session 2: Topic B'
  ],
  asyncWork: ['Task 1', 'Task 2'],
  resources: [
    { title: 'Resource Name', url: 'https://...', type: 'course' }
  ],
  exercise: {
    id: 'ex-week13',
    title: 'Exercise Title',
    description: 'What to do',
    starterCode: '# Your code here',
    solution: '# Solution'
  }
}
```

### Resource Types
- `course` 📚
- `video` 🎬
- `docs` 📖
- `tutorial` 💻
- `article` 📝
- `tool` 🔧
- `paper` 📄

---

## 📊 Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `profiles` | User info (name, email, role) |
| `progress` | Week completion tracking |
| `exercise_attempts` | Code submissions |
| `time_logs` | Time spent on platform |

### Roles
- `student` - Default role, can see own data
- `admin` - Can see all student data

---

## 🔒 Security

- Row Level Security (RLS) enabled on all tables
- Students can only see their own data
- Admins can view all student data
- API keys are public-safe (anon key only)

---

## 🆘 Troubleshooting

### "Invalid API Key" Error
- Check that `.env` variables are correct
- Make sure you're using the **anon** key, not the service key
- Restart dev server after changing `.env`

### Google Sign-In Not Working
- Verify redirect URI in Google Console matches Supabase
- Check that Google provider is enabled in Supabase
- Try incognito mode to clear cached auth state

### Admin Dashboard Not Visible
- Confirm your profile role is set to `admin` in database
- Run the SQL update command in Step 7
- Sign out and sign back in

### Progress Not Saving
- Check browser console for errors
- Verify Supabase connection in Network tab
- Check RLS policies are correctly set up

---

## 📈 Future Enhancements

Ideas for extending the app:
- [ ] Quiz system with scoring
- [ ] Discussion forums per week
- [ ] Certificate generation on completion
- [ ] Cohort management (multiple classes)
- [ ] Slack/Discord integration
- [ ] Video lesson embedding
- [ ] Peer code review

---

## 📝 License

MIT License - Use freely for your own bootcamp!

---

Built with ❤️ for AI education

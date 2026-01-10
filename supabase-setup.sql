-- ============================================
-- SUPABASE SETUP FOR NEURON ML PLATFORM
-- Run this in: Supabase Dashboard > SQL Editor
-- ============================================

-- 1. USER PROGRESS TABLE
-- Tracks lesson completion per user
create table if not exists user_progress (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  course_id text not null,
  lesson_id text not null,
  completed boolean default true,
  completed_at timestamptz default now(),
  unique(user_id, course_id, lesson_id)
);

-- 2. QUIZ SCORES TABLE
-- Stores quiz attempts and scores
create table if not exists quiz_scores (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  course_id text not null,
  lesson_id text not null,
  score integer not null,
  total integer not null,
  completed_at timestamptz default now()
);

-- 3. USER NOTES TABLE
-- Personal notes per lesson
create table if not exists user_notes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  course_id text not null,
  lesson_id text not null,
  content text,
  updated_at timestamptz default now(),
  unique(user_id, course_id, lesson_id)
);

-- 4. BOOKMARKS TABLE
-- Saved lessons
create table if not exists bookmarks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null,
  course_id text not null,
  lesson_id text not null,
  created_at timestamptz default now(),
  unique(user_id, course_id, lesson_id)
);

-- 5. USER STATS TABLE
-- XP, streaks, achievements
create table if not exists user_stats (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade not null unique,
  xp integer default 0,
  level integer default 1,
  lessons_completed integer default 0,
  quizzes_passed integer default 0,
  perfect_quizzes integer default 0,
  current_streak integer default 0,
  longest_streak integer default 0,
  last_activity_date date,
  achievements jsonb default '[]'::jsonb,
  updated_at timestamptz default now()
);

-- ============================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- Users can only access their own data
-- ============================================

-- Enable RLS on all tables
alter table user_progress enable row level security;
alter table quiz_scores enable row level security;
alter table user_notes enable row level security;
alter table bookmarks enable row level security;
alter table user_stats enable row level security;

-- user_progress policies
create policy "Users can view own progress"
  on user_progress for select
  using (auth.uid() = user_id);

create policy "Users can insert own progress"
  on user_progress for insert
  with check (auth.uid() = user_id);

create policy "Users can update own progress"
  on user_progress for update
  using (auth.uid() = user_id);

-- quiz_scores policies
create policy "Users can view own quiz scores"
  on quiz_scores for select
  using (auth.uid() = user_id);

create policy "Users can insert own quiz scores"
  on quiz_scores for insert
  with check (auth.uid() = user_id);

-- user_notes policies
create policy "Users can view own notes"
  on user_notes for select
  using (auth.uid() = user_id);

create policy "Users can insert own notes"
  on user_notes for insert
  with check (auth.uid() = user_id);

create policy "Users can update own notes"
  on user_notes for update
  using (auth.uid() = user_id);

create policy "Users can delete own notes"
  on user_notes for delete
  using (auth.uid() = user_id);

-- bookmarks policies
create policy "Users can view own bookmarks"
  on bookmarks for select
  using (auth.uid() = user_id);

create policy "Users can insert own bookmarks"
  on bookmarks for insert
  with check (auth.uid() = user_id);

create policy "Users can delete own bookmarks"
  on bookmarks for delete
  using (auth.uid() = user_id);

-- user_stats policies
create policy "Users can view own stats"
  on user_stats for select
  using (auth.uid() = user_id);

create policy "Users can insert own stats"
  on user_stats for insert
  with check (auth.uid() = user_id);

create policy "Users can update own stats"
  on user_stats for update
  using (auth.uid() = user_id);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

create index if not exists idx_user_progress_user_id on user_progress(user_id);
create index if not exists idx_user_progress_course on user_progress(user_id, course_id);
create index if not exists idx_quiz_scores_user_id on quiz_scores(user_id);
create index if not exists idx_user_notes_user_id on user_notes(user_id);
create index if not exists idx_bookmarks_user_id on bookmarks(user_id);

-- ============================================
-- FUNCTION: Auto-create user_stats on signup
-- ============================================

create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.user_stats (user_id)
  values (new.id);
  return new;
end;
$$ language plpgsql security definer;

-- Trigger to auto-create stats for new users
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

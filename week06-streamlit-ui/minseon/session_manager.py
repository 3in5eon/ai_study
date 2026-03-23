import os
from datetime import datetime
from uuid import uuid4
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

class SessionManager:
    def create(self, name=None):
        sid = uuid4().hex[:8]
        sb.table("sessions").insert({
            "id": sid,
            "name": name or "새 대화",
            "messages": [],
            "conversation": [],
            "total_cost_usd": 0.0,
        }).execute()
        return sid

    def get(self, sid):
        res = sb.table("sessions").select("*").eq("id", sid).execute()
        return res.data[0] if res.data else None

    def list(self):
        res = sb.table("sessions").select("*").order("created_at", desc=True).execute()
        return res.data

    def rename(self, sid, name):
        sb.table("sessions").update({"name": name}).eq("id", sid).execute()

    def delete(self, sid):
        sb.table("sessions").delete().eq("id", sid).execute()

    def save_messages(self, sid, messages, conversation):
        sb.table("sessions").update({
            "messages": messages,
            "conversation": conversation,
        }).eq("id", sid).execute()

    def add_cost(self, sid, cost_usd, tokens):
        session = self.get(sid)
        if session:
            sb.table("sessions").update({
                "total_cost_usd": session.get("total_cost_usd", 0) + cost_usd,
            }).eq("id", sid).execute()

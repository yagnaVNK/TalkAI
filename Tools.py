from datetime import datetime
import pytz

class AITools:
    @staticmethod
    def get_current_time():
        """Get the current time in current time zone"""
        local_tz = datetime.now().astimezone().tzinfo
        
        now = datetime.now().astimezone(local_tz)  # Local time with timezone
        return now.strftime('%I:%M %p')
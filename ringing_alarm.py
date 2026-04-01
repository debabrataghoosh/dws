
# coding: utf-8

# In[ ]:


import platform
import subprocess

def select_alarm(result) :
    if result == 0:
        sound_alarm("power_alarm.wav")
    elif result == 1 :
        sound_alarm("nomal_alarm.wav")
    else :
        sound_alarm("short_alarm.mp3")

def sound_alarm(path) :
    try:
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        return
    except Exception:
        pass

    # Fallback for environments where pygame is not installed.
    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
        except Exception:
            pass
    else:
        print("Alarm:", path)
    


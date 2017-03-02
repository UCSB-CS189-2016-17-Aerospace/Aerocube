import subprocess
import shlex

class Power():
  
  def __init__(self):
    self._power_now = "awk '{print $1*10^-6 \" W\"}' /sys/class/power_supply/BAT0/power_now"
  
  def power_now(self):
    """
    Power consumption returned in microwatts (µW)
    Look in `/sys/sys/power_supply/BAT0/power_now`
    cat /sys/class/power_supply/BAT0/power_now
    awk '{print $1*10^-6 " W"}' /sys/class/power_supply/BAT0/power_now
    """
    proc = subprocess.Popen(self._power_now, stdout=PIPE) # Might need to set shell=True
    out, err = proc.communicate()
    return out
    # may need to try proc.poll

  def power_now2(self):
    args = shlex.split(self._power_now)
    out = subprocess.check_output(args)
    return out

  def batter_capacity_in_percent(self):
    cmd = "awk '{print $1 \"%\"}' /sys/class/power_supply/BAT0/capacity"
    proc = subprocess.Popen(cmd, stdout=PIPE) # Might need to set shell=True
    out, err = proc.communicate()
    return out

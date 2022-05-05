import datetime as dt

def period(delta):
    pattern = "{h:02d}:{m:02d}:{s:02d}"
    d = {}
    d['h'], rem = divmod(delta.seconds, 3600)
    d['m'], d['s'] = divmod(rem, 60)
    d['h'] += delta.days*24
    return pattern.format(**d)

a = dt.timedelta(hours=4)
b = dt.timedelta(hours=44)

print(period(a))

print(period(b))
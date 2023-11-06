import pandas as pd
import datetime

def get_time_interval(time):
    y = pd.to_datetime(time)[0].year
    m = pd.to_datetime(time)[0].month
    d = pd.to_datetime(time)[0].day

    start1 = datetime.datetime(y,m,d,0,0,0)
    end1   = datetime.datetime(y,m,d,12,0,0)

    start2 = datetime.datetime(y,m,d,12,0,0)
    end2   = datetime.datetime(y,m,d+1,0,0,0)

    return start1, end1, start2, end2
def get_date_str(time):
    current_date = pd.Timestamp(f'{pd.to_datetime(time.data[0]).year}-{pd.to_datetime(time.data[0]).month}-{pd.to_datetime(time.data[0]).day}')
    ymd = 10000*current_date.year + 100*current_date.month + current_date.day
    return ymd

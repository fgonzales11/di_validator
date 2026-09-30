Select * from IntervalData
where deviceid in ({devices})
and valuedate >= {start} and valuedate <= {end}